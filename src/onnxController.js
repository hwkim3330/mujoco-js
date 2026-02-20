/**
 * ONNX-based Robot Controller for OpenDuck Mini
 * Ported from mujoco-web/src/onnxController.js
 *
 * Control loop matches Python EXACTLY:
 *   1. mj_step (physics)
 *   2. counter++
 *   3. if counter % decimation == 0:
 *        a. update imitation phase
 *        b. obs = get_obs()        (uses OLD action history, OLD motor_targets)
 *        c. action = policy(obs)    (synchronous via await)
 *        d. update action history   (AFTER obs)
 *        e. motor_targets = default + action * scale
 *        f. velocity clamp
 *        g. data.ctrl = motor_targets
 */

export class OnnxController {
  constructor(mujoco, model, data) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.session = null;
    this.enabled = false;

    this.actionScale = 0.25;
    this.dofVelScale = 0.05;
    this.maxMotorVelocity = 5.24;
    this.simDt = 0.002;
    this.decimation = 10;
    this.ctrlDt = this.simDt * this.decimation;

    this.numDofs = 14;

    this.lastAction = null;
    this.lastLastAction = null;
    this.lastLastLastAction = null;
    this.motorTargets = null;
    this.prevMotorTargets = null;
    this.defaultActuator = null;

    this.commands = [0, 0, 0, 0, 0, 0, 0];
    this.defaultForwardCommand = 0.03;
    this.defaultNeckPitchCommand = 0.0;
    this.startupNeckPitchCommand = 0.05;
    this.startupAssistDuration = 0.8;

    this.imitationI = 0;
    this.nbStepsInPeriod = 27;
    this.imitationPhase = [0, 0];

    this.stepCounter = 0;
    this.policyStepCount = 0;
    this._policyRunning = false;

    this.gyroAddr = -1;
    this.accelAddr = -1;

    this.leftFootBodyId = -1;
    this.rightFootBodyId = -1;
    this.floorBodyId = -1;

    this.qposIndices = null;
    this.qvelIndices = null;
  }

  async loadModel(url) {
    try {
      if (typeof ort === 'undefined') {
        console.warn('ONNX Runtime not loaded.');
        return false;
      }
      this.session = await ort.InferenceSession.create(url);
      console.log('ONNX model loaded:', url);

      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];

      this.numDofs = this.model.nu;
      this.initState();
      this.findSensorAddresses();
      this.findBodyIds();
      this.findJointIndices();
      return true;
    } catch (e) {
      console.error('Failed to load ONNX model:', e);
      return false;
    }
  }

  initState() {
    const n = this.numDofs;
    this.lastAction = new Float32Array(n);
    this.lastLastAction = new Float32Array(n);
    this.lastLastLastAction = new Float32Array(n);
    this.motorTargets = new Float32Array(n);
    this.prevMotorTargets = new Float32Array(n);
    this.defaultActuator = new Float32Array(n);

    if (this.model.nkey > 0 && this.model.key_ctrl) {
      for (let i = 0; i < n; i++) {
        this.defaultActuator[i] = this.model.key_ctrl[i] || 0;
      }
    } else {
      const homeCtrl = [
        0.002, 0.053, -0.63, 1.368, -0.784,
        0, 0, 0, 0,
        -0.003, -0.065, 0.635, 1.379, -0.796
      ];
      for (let i = 0; i < n; i++) {
        this.defaultActuator[i] = homeCtrl[i] || 0;
      }
    }

    for (let i = 0; i < n; i++) {
      this.motorTargets[i] = this.defaultActuator[i];
      this.prevMotorTargets[i] = this.defaultActuator[i];
    }

    const ctrl = this.data.ctrl;
    for (let i = 0; i < Math.min(n, ctrl.length); i++) {
      ctrl[i] = this.motorTargets[i];
    }

    this.commands[0] = this.defaultForwardCommand;
    this.commands[3] = this.startupNeckPitchCommand;
  }

  findSensorAddresses() {
    const nsensor = this.model.nsensor;
    const names = this.model.names;

    const getNameAt = (nameAdr) => {
      if (!names || nameAdr < 0 || nameAdr >= names.length) return '';
      let name = '';
      for (let j = nameAdr; j < names.length && names[j] !== 0; j++) {
        name += String.fromCharCode(names[j]);
      }
      return name;
    };

    if (names && this.model.name_sensoradr) {
      for (let i = 0; i < nsensor; i++) {
        const nameAdr = this.model.name_sensoradr[i];
        const sensorName = getNameAt(nameAdr);
        const adr = this.model.sensor_adr[i];
        if (sensorName === 'gyro') this.gyroAddr = adr;
        if (sensorName === 'accelerometer') this.accelAddr = adr;
      }
    }

    if (this.gyroAddr < 0 || this.accelAddr < 0) {
      for (let i = 0; i < nsensor; i++) {
        const type = this.model.sensor_type[i];
        const adr = this.model.sensor_adr[i];
        if (type === 3 && this.gyroAddr < 0) this.gyroAddr = adr;
        if (type === 1 && this.accelAddr < 0) this.accelAddr = adr;
      }
    }

    if (this.gyroAddr < 0) this.gyroAddr = 0;
    if (this.accelAddr < 0) this.accelAddr = 6;
  }

  findBodyIds() {
    try {
      this.leftFootBodyId = this.mujoco.mj_name2id(this.model, 1, 'foot_assembly');
      this.rightFootBodyId = this.mujoco.mj_name2id(this.model, 1, 'foot_assembly_2');
      this.floorBodyId = this.mujoco.mj_name2id(this.model, 1, 'floor');
    } catch (e) {
      console.warn('Could not find body IDs for contact detection:', e);
    }
  }

  findJointIndices() {
    const jointNames = [
      'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
      'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
      'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle'
    ];

    this.qposIndices = new Int32Array(this.numDofs);
    this.qvelIndices = new Int32Array(this.numDofs);

    let success = false;
    try {
      if (this.model.jnt_qposadr && this.model.jnt_dofadr) {
        for (let i = 0; i < this.numDofs; i++) {
          const jointId = this.mujoco.mj_name2id(this.model, 3, jointNames[i]);
          if (jointId >= 0) {
            this.qposIndices[i] = this.model.jnt_qposadr[jointId];
            this.qvelIndices[i] = this.model.jnt_dofadr[jointId];
          } else {
            throw new Error(`Joint "${jointNames[i]}" not found`);
          }
        }
        success = true;
      }
    } catch (e) {
      console.warn('Joint index lookup failed, using fallback:', e);
    }

    if (!success) {
      for (let i = 0; i < this.numDofs; i++) {
        this.qposIndices[i] = 7 + i;
        this.qvelIndices[i] = 6 + i;
      }
    }
  }

  setCommand(linX, linY, angZ) {
    this.commands[0] = Math.max(-0.15, Math.min(0.15, linX));
    this.commands[1] = Math.max(-0.2, Math.min(0.2, linY));
    this.commands[2] = Math.max(-1.0, Math.min(1.0, angZ));
  }

  getObservation() {
    const obs = [];
    const sd = this.data.sensordata;

    // Gyro (3)
    obs.push(sd[this.gyroAddr], sd[this.gyroAddr + 1], sd[this.gyroAddr + 2]);

    // Accelerometer (3) with bias
    obs.push(sd[this.accelAddr] + 1.3, sd[this.accelAddr + 1], sd[this.accelAddr + 2]);

    // Commands (7)
    obs.push(...this.commands);

    // Joint angles - default (numDofs)
    for (let i = 0; i < this.numDofs; i++) {
      obs.push(this.data.qpos[this.qposIndices[i]] - this.defaultActuator[i]);
    }

    // Joint velocities * scale (numDofs)
    for (let i = 0; i < this.numDofs; i++) {
      obs.push(this.data.qvel[this.qvelIndices[i]] * this.dofVelScale);
    }

    // Last 3 actions (numDofs * 3)
    obs.push(...this.lastAction);
    obs.push(...this.lastLastAction);
    obs.push(...this.lastLastLastAction);

    // Motor targets (numDofs)
    obs.push(...this.motorTargets);

    // Foot contacts (2)
    obs.push(...this.getFeetContacts());

    // Imitation phase (2)
    obs.push(...this.imitationPhase);

    return new Float32Array(obs);
  }

  getFeetContacts() {
    if (this.leftFootBodyId >= 0 && this.rightFootBodyId >= 0 && this.floorBodyId >= 0) {
      let leftContact = 0.0;
      let rightContact = 0.0;

      const ncon = this.data.ncon;
      for (let i = 0; i < ncon; i++) {
        try {
          const contact = this.data.contact.get(i);
          if (!contact) continue;
          const body1 = this.model.geom_bodyid[contact.geom1];
          const body2 = this.model.geom_bodyid[contact.geom2];

          if ((body1 === this.leftFootBodyId && body2 === this.floorBodyId) ||
              (body1 === this.floorBodyId && body2 === this.leftFootBodyId)) {
            leftContact = 1.0;
          }
          if ((body1 === this.rightFootBodyId && body2 === this.floorBodyId) ||
              (body1 === this.floorBodyId && body2 === this.rightFootBodyId)) {
            rightContact = 1.0;
          }
        } catch (e) {
          break;
        }
      }
      return [leftContact, rightContact];
    }

    const height = this.data.qpos[2] || 0;
    return height < 0.18 ? [1.0, 1.0] : [0.0, 0.0];
  }

  async runPolicy() {
    if (!this.session || !this.enabled) return;

    this.policyStepCount++;

    // Startup assist: blend neck pitch (only during first 0.8s, then user controls freely)
    const simTime = this.stepCounter * this.simDt;
    if (simTime < this.startupAssistDuration) {
      const a = simTime / this.startupAssistDuration;
      const neckTarget = this.startupNeckPitchCommand * (1 - a) + this.defaultNeckPitchCommand * a;
      this.commands[3] = Math.max(this.commands[3], neckTarget);
    }

    // 1. Update imitation phase
    this.imitationI = (this.imitationI + 1) % this.nbStepsInPeriod;
    const phase = (this.imitationI / this.nbStepsInPeriod) * 2 * Math.PI;
    this.imitationPhase[0] = Math.cos(phase);
    this.imitationPhase[1] = Math.sin(phase);

    // 2. Build observation
    const obs = this.getObservation();
    if (obs.some(v => isNaN(v))) {
      console.warn(`[Policy #${this.policyStepCount}] NaN in observation! Skipping.`);
      return;
    }

    // Diagnostics for first policy steps
    if (this.policyStepCount <= 5) {
      const h = (this.data.qpos[2] || 0).toFixed(4);
      const contacts = this.getFeetContacts();
      const gyro = [this.data.sensordata[this.gyroAddr], this.data.sensordata[this.gyroAddr+1], this.data.sensordata[this.gyroAddr+2]];
      console.log(`[Policy #${this.policyStepCount}] H=${h} contacts=[${contacts}] gyro=[${gyro.map(v=>v.toFixed(3))}] obs.len=${obs.length} cmd=[${this.commands.slice(0,3).map(v=>v.toFixed(3))}]`);
    }

    // 3. Run ONNX inference
    try {
      const inputTensor = new ort.Tensor('float32', obs, [1, obs.length]);
      const feeds = {};
      feeds[this.inputName] = inputTensor;

      const results = await this.session.run(feeds);
      const output = results[this.outputName];
      if (!output) return;

      const action = new Float32Array(output.data);

      if (this.policyStepCount <= 5) {
        console.log(`  action=[${Array.from(action).slice(0,5).map(v=>v.toFixed(3))}...] range=[${Math.min(...action).toFixed(3)}, ${Math.max(...action).toFixed(3)}]`);
      }

      // 4. Update action history AFTER inference
      this.lastLastLastAction.set(this.lastLastAction);
      this.lastLastAction.set(this.lastAction);
      this.lastAction.set(action);

      // 5. Compute new motor targets
      for (let i = 0; i < this.numDofs; i++) {
        this.motorTargets[i] = this.defaultActuator[i] + action[i] * this.actionScale;
      }

      // 5b. Direct head control: override head joints when user input present
      // Joint indices: 5=neck_pitch, 7=head_yaw
      // Policy observation still includes commands so legs compensate for balance
      if (Math.abs(this.commands[3]) > 0.05) {
        this.motorTargets[5] = this.defaultActuator[5] + this.commands[3];
      }
      if (Math.abs(this.commands[5]) > 0.05) {
        this.motorTargets[7] = this.defaultActuator[7] + this.commands[5];
      }

      // 6. Velocity clamp
      for (let i = 0; i < this.numDofs; i++) {
        const maxChange = this.maxMotorVelocity * this.ctrlDt;
        const diff = this.motorTargets[i] - this.prevMotorTargets[i];
        if (Math.abs(diff) > maxChange) {
          this.motorTargets[i] = this.prevMotorTargets[i] + Math.sign(diff) * maxChange;
        }
        this.prevMotorTargets[i] = this.motorTargets[i];
      }

      // 7. Apply to ctrl
      const ctrl = this.data.ctrl;
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.motorTargets[i];
      }
    } catch (e) {
      console.error('ONNX inference error:', e);
    }
  }

  runPolicyAsync() {
    if (this._policyRunning) return;
    this._policyRunning = true;
    this.runPolicy().then(() => {
      this._policyRunning = false;
    }).catch((e) => {
      console.error('Policy error:', e);
      this._policyRunning = false;
    });
  }

  reset() {
    if (this.lastAction) this.lastAction.fill(0);
    if (this.lastLastAction) this.lastLastAction.fill(0);
    if (this.lastLastLastAction) this.lastLastLastAction.fill(0);

    this.imitationI = 0;
    this.imitationPhase = [0, 0];
    this.commands = [this.defaultForwardCommand, 0, 0, this.startupNeckPitchCommand, 0, 0, 0];
    this.stepCounter = 0;
    this.policyStepCount = 0;

    if (this.motorTargets && this.defaultActuator) {
      this.motorTargets.set(this.defaultActuator);
      this.prevMotorTargets.set(this.defaultActuator);
      const ctrl = this.data.ctrl;
      for (let i = 0; i < Math.min(this.numDofs, ctrl.length); i++) {
        ctrl[i] = this.defaultActuator[i];
      }
    }
    this._policyRunning = false;
  }
}
