/**
 * CPG (Central Pattern Generator) walking controller for Unitree H1.
 * Procedural gait — no ONNX policy needed.
 *
 * H1 actuator layout (19 torque motors):
 *  0: left_hip_yaw      5: right_hip_yaw     10: torso
 *  1: left_hip_roll      6: right_hip_roll     11: left_shoulder_pitch
 *  2: left_hip_pitch     7: right_hip_pitch    12: left_shoulder_roll
 *  3: left_knee          8: right_knee         13: left_shoulder_yaw
 *  4: left_ankle         9: right_ankle        14: left_elbow
 *                                              15: right_shoulder_pitch
 *                                              16: right_shoulder_roll
 *                                              17: right_shoulder_yaw
 *                                              18: right_elbow
 *
 * Home keyframe qpos (after free joint 7-DOF):
 *  left_leg:  [0, 0, -0.4, 0.8, -0.4]
 *  right_leg: [0, 0, -0.4, 0.8, -0.4]
 *  torso: 0
 *  left_arm:  [0, 0, 0, 0]
 *  right_arm: [0, 0, 0, 0]
 */

export class CpgController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.enabled = false;

    this.simDt = model.opt.timestep || 0.002;

    // Gait parameters
    this.frequency = 1.2;        // Hz (steps per second)
    this.phase = 0;              // Phase oscillator [0, 2*PI)

    // Amplitudes for leg joints
    this.hipPitchAmp = 0.25;     // Forward swing amplitude
    this.kneeAmp = 0.35;         // Knee bend during swing
    this.ankleAmp = 0.15;        // Ankle compensation
    this.hipRollAmp = 0.03;      // Lateral balance

    // Arm swing
    this.armSwingGain = 0.3;

    // PD gains for balance (applied as torque corrections)
    this.balanceKp = 50.0;       // Proportional gain for trunk lean correction
    this.balanceKd = 5.0;        // Derivative gain

    // Home pose (joint targets at stance)
    this.homeQpos = {
      left_hip_yaw: 0, left_hip_roll: 0, left_hip_pitch: -0.4, left_knee: 0.8, left_ankle: -0.4,
      right_hip_yaw: 0, right_hip_roll: 0, right_hip_pitch: -0.4, right_knee: 0.8, right_ankle: -0.4,
      torso: 0,
      left_shoulder_pitch: 0, left_shoulder_roll: 0.2, left_shoulder_yaw: 0, left_elbow: -0.3,
      right_shoulder_pitch: 0, right_shoulder_roll: -0.2, right_shoulder_yaw: 0, right_elbow: -0.3,
    };

    // Actuator index map (from H1 XML actuator order)
    this.actIdx = {
      left_hip_yaw: 0, left_hip_roll: 1, left_hip_pitch: 2, left_knee: 3, left_ankle: 4,
      right_hip_yaw: 5, right_hip_roll: 6, right_hip_pitch: 7, right_knee: 8, right_ankle: 9,
      torso: 10,
      left_shoulder_pitch: 11, left_shoulder_roll: 12, left_shoulder_yaw: 13, left_elbow: 14,
      right_shoulder_pitch: 15, right_shoulder_roll: 16, right_shoulder_yaw: 17, right_elbow: 18,
    };

    // Joint qpos indices (after freejoint: offset 7)
    this.jntIdx = {};
    this.findJointIndices();

    // Commands
    this.forwardSpeed = 0.5;     // Target forward velocity (0-1 scale)
    this.lateralSpeed = 0;
    this.turnRate = 0;

    // State for balance
    this.prevPitch = 0;
    this.prevRoll = 0;
  }

  findJointIndices() {
    const names = Object.keys(this.actIdx);
    for (const name of names) {
      try {
        const jid = this.mujoco.mj_name2id(this.model, 3, name); // mjOBJ_JOINT=3
        if (jid >= 0 && this.model.jnt_qposadr) {
          this.jntIdx[name] = this.model.jnt_qposadr[jid];
        }
      } catch (e) {
        // Will use fallback
      }
    }
  }

  setCommand(forward, lateral, turn) {
    this.forwardSpeed = Math.max(-1, Math.min(1, forward));
    this.lateralSpeed = Math.max(-0.5, Math.min(0.5, lateral));
    this.turnRate = Math.max(-1, Math.min(1, turn));
  }

  /**
   * Extract trunk orientation from qpos quaternion.
   * Returns { pitch, roll } in radians.
   */
  getTrunkOrientation() {
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];

    // Euler angles from quaternion (ZYX convention)
    const sinp = 2 * (qw * qy - qz * qx);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
    const roll = Math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));

    return { pitch, roll };
  }

  /**
   * Compute PD torque for a joint to track a target position.
   * H1 uses torque actuators, so we compute: tau = kp*(target - q) - kd*qdot
   */
  pdTorque(jointName, target, kp, kd) {
    const idx = this.jntIdx[jointName];
    if (idx === undefined) return 0;

    // Find qvel index via jnt_dofadr
    let dofIdx;
    try {
      const jid = this.mujoco.mj_name2id(this.model, 3, jointName);
      dofIdx = this.model.jnt_dofadr[jid];
    } catch (e) {
      dofIdx = idx - 7 + 6; // fallback
    }

    const q = this.data.qpos[idx];
    const qdot = this.data.qvel[dofIdx] || 0;

    return kp * (target - q) - kd * qdot;
  }

  /**
   * Step the CPG and compute torques.
   * Called every physics step.
   */
  step() {
    if (!this.enabled) return;

    // Advance phase
    this.phase += 2 * Math.PI * this.frequency * this.simDt;
    if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;

    const leftPhase = this.phase;
    const rightPhase = this.phase + Math.PI; // Opposite leg

    // Scale amplitudes by forward command
    const ampScale = Math.abs(this.forwardSpeed);
    const direction = Math.sign(this.forwardSpeed) || 1;

    // Get trunk orientation for balance
    const { pitch, roll } = this.getTrunkOrientation();
    const pitchRate = (pitch - this.prevPitch) / this.simDt;
    const rollRate = (roll - this.prevRoll) / this.simDt;
    this.prevPitch = pitch;
    this.prevRoll = roll;

    // PD gains for each joint type
    const hipKp = 150, hipKd = 8;
    const kneeKp = 200, kneeKd = 10;
    const ankleKp = 30, ankleKd = 2;
    const torsoKp = 150, torsoKd = 10;
    const armKp = 20, armKd = 2;

    const ctrl = this.data.ctrl;

    // --- LEFT LEG ---
    const leftSwing = Math.sin(leftPhase);
    const leftStance = Math.max(0, -Math.sin(leftPhase)); // Positive during stance

    const leftHipPitchTarget = this.homeQpos.left_hip_pitch
      + direction * this.hipPitchAmp * ampScale * leftSwing;
    const leftKneeTarget = this.homeQpos.left_knee
      + this.kneeAmp * ampScale * Math.max(0, Math.sin(leftPhase));
    const leftAnkleTarget = this.homeQpos.left_ankle
      - this.ankleAmp * ampScale * leftSwing;
    const leftHipRollTarget = this.homeQpos.left_hip_roll
      - this.hipRollAmp * leftStance  // Lean into stance leg
      + this.lateralSpeed * 0.05;
    const leftHipYawTarget = this.homeQpos.left_hip_yaw
      + this.turnRate * 0.05 * leftSwing;

    ctrl[this.actIdx.left_hip_yaw] = this.pdTorque('left_hip_yaw', leftHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_roll] = this.pdTorque('left_hip_roll', leftHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_hip_pitch] = this.pdTorque('left_hip_pitch', leftHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.left_knee] = this.pdTorque('left_knee', leftKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.left_ankle] = this.pdTorque('left_ankle', leftAnkleTarget, ankleKp, ankleKd);

    // --- RIGHT LEG ---
    const rightSwing = Math.sin(rightPhase);
    const rightStance = Math.max(0, -Math.sin(rightPhase));

    const rightHipPitchTarget = this.homeQpos.right_hip_pitch
      + direction * this.hipPitchAmp * ampScale * rightSwing;
    const rightKneeTarget = this.homeQpos.right_knee
      + this.kneeAmp * ampScale * Math.max(0, Math.sin(rightPhase));
    const rightAnkleTarget = this.homeQpos.right_ankle
      - this.ankleAmp * ampScale * rightSwing;
    const rightHipRollTarget = this.homeQpos.right_hip_roll
      + this.hipRollAmp * rightStance
      + this.lateralSpeed * 0.05;
    const rightHipYawTarget = this.homeQpos.right_hip_yaw
      + this.turnRate * 0.05 * rightSwing;

    ctrl[this.actIdx.right_hip_yaw] = this.pdTorque('right_hip_yaw', rightHipYawTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_roll] = this.pdTorque('right_hip_roll', rightHipRollTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_hip_pitch] = this.pdTorque('right_hip_pitch', rightHipPitchTarget, hipKp, hipKd);
    ctrl[this.actIdx.right_knee] = this.pdTorque('right_knee', rightKneeTarget, kneeKp, kneeKd);
    ctrl[this.actIdx.right_ankle] = this.pdTorque('right_ankle', rightAnkleTarget, ankleKp, ankleKd);

    // --- TORSO ---
    // Keep torso upright, with yaw control
    const torsoTarget = this.homeQpos.torso + this.turnRate * 0.1;
    ctrl[this.actIdx.torso] = this.pdTorque('torso', torsoTarget, torsoKp, torsoKd);

    // --- BALANCE CORRECTIONS ---
    // Add pitch correction to hip pitch (lean forward/back → adjust hip)
    const pitchCorrection = -this.balanceKp * pitch - this.balanceKd * pitchRate;
    ctrl[this.actIdx.left_hip_pitch] += pitchCorrection * 0.3;
    ctrl[this.actIdx.right_hip_pitch] += pitchCorrection * 0.3;

    // Add roll correction to hip roll
    const rollCorrection = -this.balanceKp * roll - this.balanceKd * rollRate;
    ctrl[this.actIdx.left_hip_roll] += rollCorrection * 0.2;
    ctrl[this.actIdx.right_hip_roll] -= rollCorrection * 0.2;

    // --- ARMS (opposite to legs) ---
    const leftArmSwing = -this.armSwingGain * direction * ampScale * leftSwing;
    const rightArmSwing = -this.armSwingGain * direction * ampScale * rightSwing;

    ctrl[this.actIdx.left_shoulder_pitch] = this.pdTorque('left_shoulder_pitch',
      this.homeQpos.left_shoulder_pitch + leftArmSwing, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_roll] = this.pdTorque('left_shoulder_roll',
      this.homeQpos.left_shoulder_roll, armKp, armKd);
    ctrl[this.actIdx.left_shoulder_yaw] = this.pdTorque('left_shoulder_yaw',
      this.homeQpos.left_shoulder_yaw, armKp * 0.5, armKd);
    ctrl[this.actIdx.left_elbow] = this.pdTorque('left_elbow',
      this.homeQpos.left_elbow, armKp, armKd);

    ctrl[this.actIdx.right_shoulder_pitch] = this.pdTorque('right_shoulder_pitch',
      this.homeQpos.right_shoulder_pitch + rightArmSwing, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_roll] = this.pdTorque('right_shoulder_roll',
      this.homeQpos.right_shoulder_roll, armKp, armKd);
    ctrl[this.actIdx.right_shoulder_yaw] = this.pdTorque('right_shoulder_yaw',
      this.homeQpos.right_shoulder_yaw, armKp * 0.5, armKd);
    ctrl[this.actIdx.right_elbow] = this.pdTorque('right_elbow',
      this.homeQpos.right_elbow, armKp, armKd);

    // Clamp all controls to their ranges
    if (this.model.actuator_ctrlrange) {
      for (let i = 0; i < this.model.nu; i++) {
        const lo = this.model.actuator_ctrlrange[i * 2];
        const hi = this.model.actuator_ctrlrange[i * 2 + 1];
        ctrl[i] = Math.max(lo, Math.min(hi, ctrl[i]));
      }
    }
  }

  reset() {
    this.phase = 0;
    this.prevPitch = 0;
    this.prevRoll = 0;
  }
}
