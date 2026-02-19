/**
 * main.js — MuJoCo WASM + Three.js playground with walking controllers.
 * Supports: OpenDuck (ONNX), Unitree H1 (CPG).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import load_mujoco from 'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js';
import { buildScene, getPosition, getQuaternion } from './meshBuilder.js';
import { loadSceneAssets } from './assetLoader.js';
import { OnnxController } from './onnxController.js';
import { CpgController } from './cpgController.js';

// ─── DOM ────────────────────────────────────────────────────────────
const statusEl = document.getElementById('status');
const sceneSelect = document.getElementById('scene-select');
const resetBtn = document.getElementById('btn-reset');
const controllerBtn = document.getElementById('btn-controller');
const helpOverlay = document.getElementById('help-overlay');

// ─── Three.js Setup ─────────────────────────────────────────────────
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11151d);
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.0));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(3, 5, 3);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(3.0, 2.0, 3.0);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;

// ─── State ──────────────────────────────────────────────────────────
let mujoco;
let model;
let data;
let bodies = {};
let mujocoRoot = null;

let onnxController = null;
let cpgController = null;
let activeController = null; // 'onnx' | 'cpg' | null

let paused = false;
let cameraFollow = true;

// Keyboard state
const keys = {};

// Touch / joystick state
let touchX = 0; // -1..1 (left/right)
let touchY = 0; // -1..1 (back/forward)
let touchRotL = false;
let touchRotR = false;
let headX = 0;
let headY = 0;
const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

// Step counter for ONNX decimation
let stepCounter = 0;

// ─── Obstacle System ─────────────────────────────────────────────────
const NUM_BALLS = 5;
const NUM_BOXES = 5;
const NUM_OBSTACLES = NUM_BALLS + NUM_BOXES;
const HIDE_Z = -50;

let obstacleQposBase = -1;
let obstacleQvelBase = -1;
let nextBall = 0;
let nextBox = 0;
let currentObstacleScale = 1;

// ─── Scene Config ───────────────────────────────────────────────────
const SCENES = {
  'unitree_h1/scene.xml': {
    controller: 'cpg',
    camera: { pos: [3.0, 2.0, 3.0], target: [0, 0.9, 0] },
  },
  'openduck/scene_flat_terrain.xml': {
    controller: 'onnx',
    camera: { pos: [0.5, 0.4, 0.5], target: [0, 0.15, 0] },
  },
  'openduck/scene_flat_terrain_backlash.xml': {
    controller: 'onnx',
    camera: { pos: [0.5, 0.4, 0.5], target: [0, 0.15, 0] },
  },
};

let currentScenePath = 'openduck/scene_flat_terrain_backlash.xml';

// ─── Obstacle XML Generation ────────────────────────────────────────
function generateArenaXML(sceneXml, scale) {
  let xml = sceneXml;

  const colors = [
    '0.95 0.25 0.2 1',
    '0.2 0.55 0.95 1',
    '0.2 0.85 0.35 1',
    '0.95 0.75 0.15 1',
    '0.85 0.3 0.7 1',
  ];

  let obsXml = '\n  <worldbody>\n';
  for (let i = 0; i < NUM_BALLS; i++) {
    const r = (0.015 + i * 0.004) * scale;
    const m = (0.03 * scale * scale).toFixed(3);
    obsXml += `    <body name="obstacle_${i}" pos="0 0 ${HIDE_Z}"><freejoint name="obs_fj_${i}"/><geom type="sphere" size="${r.toFixed(4)}" rgba="${colors[i]}" mass="${m}" contype="1" conaffinity="1"/></body>\n`;
  }
  for (let i = 0; i < NUM_BOXES; i++) {
    const r = (0.012 + i * 0.003) * scale;
    const m = (0.04 * scale * scale).toFixed(3);
    const idx = NUM_BALLS + i;
    obsXml += `    <body name="obstacle_${idx}" pos="0 0 ${HIDE_Z}"><freejoint name="obs_fj_${idx}"/><geom type="box" size="${r.toFixed(4)} ${r.toFixed(4)} ${r.toFixed(4)}" rgba="${colors[i]}" mass="${m}" contype="1" conaffinity="1"/></body>\n`;
  }
  obsXml += '  </worldbody>\n';

  xml = xml.replace('</mujoco>', obsXml + '</mujoco>');

  // Extend keyframe qpos (each freejoint = 7 DOFs: pos + quat)
  const obsQpos = Array(NUM_OBSTACLES).fill(`0 0 ${HIDE_Z} 1 0 0 0`).join(' ');
  xml = xml.replace(
    /(qpos\s*=\s*")([\s\S]*?)(")/,
    (m, pre, content, post) => pre + content.trimEnd() + ' ' + obsQpos + '\n    ' + post
  );

  return xml;
}

function findObstacleIndices() {
  obstacleQposBase = -1;
  obstacleQvelBase = -1;
  try {
    const bodyId = mujoco.mj_name2id(model, 1, 'obstacle_0');
    if (bodyId < 0) return;
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_bodyid[j] === bodyId) {
        obstacleQposBase = model.jnt_qposadr[j];
        obstacleQvelBase = model.jnt_dofadr[j];
        break;
      }
    }
    console.log(`Obstacles: qposBase=${obstacleQposBase}, qvelBase=${obstacleQvelBase}`);
  } catch (e) {
    console.warn('Could not find obstacle indices:', e);
  }
}

function spawnObstacle(type) {
  if (obstacleQposBase < 0 || !model || !data) return;

  let idx;
  if (type === 'box') {
    idx = NUM_BALLS + (nextBox % NUM_BOXES);
    nextBox++;
  } else {
    idx = nextBall % NUM_BALLS;
    nextBall++;
  }

  const rx = data.qpos[0];
  const ry = data.qpos[1];
  const rz = data.qpos[2];

  const angle = Math.random() * Math.PI * 2;
  const dist = (0.15 + Math.random() * 0.25) * currentObstacleScale;

  const base = obstacleQposBase + idx * 7;
  data.qpos[base + 0] = rx + Math.cos(angle) * dist;
  data.qpos[base + 1] = ry + Math.sin(angle) * dist;
  data.qpos[base + 2] = rz + 0.2 * currentObstacleScale;
  data.qpos[base + 3] = 1;
  data.qpos[base + 4] = 0;
  data.qpos[base + 5] = 0;
  data.qpos[base + 6] = 0;

  const vbase = obstacleQvelBase + idx * 6;
  for (let v = 0; v < 6; v++) data.qvel[vbase + v] = 0;

  mujoco.mj_forward(model, data);
}

// ─── Functions ──────────────────────────────────────────────────────
function setStatus(text) {
  statusEl.textContent = text;
}

function clearScene() {
  if (mujocoRoot) {
    scene.remove(mujocoRoot);
    mujocoRoot = null;
  }
  bodies = {};
}

async function loadScene(scenePath) {
  setStatus(`Loading: ${scenePath}`);

  // Load assets to VFS
  await loadSceneAssets(mujoco, scenePath, setStatus);

  // Generate arena XML with obstacles
  currentObstacleScale = scenePath.includes('unitree') ? 3.5 : 2.5;
  const originalXml = new TextDecoder().decode(mujoco.FS.readFile('/working/' + scenePath));
  const arenaXml = generateArenaXML(originalXml, currentObstacleScale);
  const arenaPath = scenePath.replace('.xml', '_arena.xml');
  mujoco.FS.writeFile('/working/' + arenaPath, arenaXml);

  // Clean up old model
  clearScene();
  if (data) { data.delete(); data = null; }
  if (model) { model.delete(); model = null; }

  // Load MuJoCo model (arena with obstacles)
  model = mujoco.MjModel.loadFromXML('/working/' + arenaPath);
  data = new mujoco.MjData(model);

  console.log(`Model loaded: nq=${model.nq}, nv=${model.nv}, nu=${model.nu}, ngeom=${model.ngeom}, nbody=${model.nbody}, nkey=${model.nkey}`);
  console.log(`Timestep: ${model.opt.timestep}, iterations: ${model.opt.iterations}`);

  // Apply home keyframe
  if (model.nkey > 0) {
    data.qpos.set(model.key_qpos.slice(0, model.nq));
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    if (model.key_ctrl) data.ctrl.set(model.key_ctrl.slice(0, model.nu));
    mujoco.mj_forward(model, data);
  }

  // Build Three.js scene from model
  const built = buildScene(model);
  mujocoRoot = built.mujocoRoot;
  bodies = built.bodies;
  scene.add(mujocoRoot);

  // Setup controller
  activeController = null;
  onnxController = null;
  cpgController = null;
  stepCounter = 0;

  const cfg = SCENES[scenePath] || {};

  if (cfg.controller === 'onnx') {
    // OpenDuck ONNX walking
    model.opt.iterations = 30; // WASM needs more solver iterations
    console.log(`Set solver iterations = ${model.opt.iterations}`);

    onnxController = new OnnxController(mujoco, model, data);
    const loaded = await onnxController.loadModel('./assets/models/openduck_walk.onnx');
    if (loaded) {
      // Warm up physics with ctrl set to default
      for (let i = 0; i < 200; i++) mujoco.mj_step(model, data);
      console.log(`Warm-up done. Height=${data.qpos[2]?.toFixed(4)}`);
      onnxController.enabled = true;
      activeController = 'onnx';
    }
  } else if (cfg.controller === 'cpg') {
    // Unitree H1 CPG walking
    cpgController = new CpgController(mujoco, model, data);
    // Warm up physics
    for (let i = 0; i < 200; i++) {
      cpgController.step(); // PD hold during warm-up
      mujoco.mj_step(model, data);
    }
    console.log(`H1 warm-up done. Height=${data.qpos[2]?.toFixed(4)}`);
    cpgController.enabled = true;
    activeController = 'cpg';
  }

  findObstacleIndices();
  nextBall = 0;
  nextBox = 0;
  updateControllerBtn();

  // Set camera
  if (cfg.camera) {
    camera.position.set(...cfg.camera.pos);
    controls.target.set(...cfg.camera.target);
  }
  controls.update();

  currentScenePath = scenePath;
  setStatus(`Ready: ${scenePath.split('/').pop()}`);
}

function updateControllerBtn() {
  if (!controllerBtn) return;
  if (activeController === 'onnx') {
    controllerBtn.textContent = onnxController?.enabled ? 'Policy: ON' : 'Policy: OFF';
    controllerBtn.style.display = '';
  } else if (activeController === 'cpg') {
    controllerBtn.textContent = cpgController?.enabled ? 'CPG: ON' : 'CPG: OFF';
    controllerBtn.style.display = '';
  } else {
    controllerBtn.style.display = 'none';
  }
}

function resetScene() {
  if (!model || !data) return;

  if (model.nkey > 0) {
    data.qpos.set(model.key_qpos.slice(0, model.nq));
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    if (model.key_ctrl) data.ctrl.set(model.key_ctrl.slice(0, model.nu));
    mujoco.mj_forward(model, data);
  }

  stepCounter = 0;
  nextBall = 0;
  nextBox = 0;

  if (onnxController) {
    onnxController.reset();
    // Re-apply ctrl after reset
    for (let i = 0; i < 200; i++) mujoco.mj_step(model, data);
    onnxController.enabled = true;
  }
  if (cpgController) {
    cpgController.reset();
    for (let i = 0; i < 200; i++) {
      cpgController.step();
      mujoco.mj_step(model, data);
    }
    cpgController.enabled = true;
  }
  updateControllerBtn();
}

function toggleController() {
  if (activeController === 'onnx' && onnxController) {
    onnxController.enabled = !onnxController.enabled;
  } else if (activeController === 'cpg' && cpgController) {
    cpgController.enabled = !cpgController.enabled;
  }
  updateControllerBtn();
}

// ─── Keyboard ───────────────────────────────────────────────────────
function handleInput() {
  // Merge keyboard + touch input
  const kbFwd = keys['KeyW'] || keys['ArrowUp'];
  const kbBack = keys['KeyS'] || keys['ArrowDown'];
  const kbLeft = keys['KeyA'] || keys['ArrowLeft'];
  const kbRight = keys['KeyD'] || keys['ArrowRight'];
  const kbRotL = keys['KeyQ'];
  const kbRotR = keys['KeyE'];

  if (activeController === 'onnx' && onnxController && onnxController.enabled) {
    let linX = onnxController.defaultForwardCommand;
    let linY = 0;
    let angZ = 0;

    // Keyboard
    if (kbFwd) linX = 0.10;
    if (kbBack) linX = -0.10;
    if (kbLeft) linY = 0.15;
    if (kbRight) linY = -0.15;
    if (kbRotL) angZ = 0.5;
    if (kbRotR) angZ = -0.5;

    // Touch joystick override (if active)
    if (Math.abs(touchY) > 0.15 || Math.abs(touchX) > 0.15) {
      linX = touchY * 0.12;  // Forward/back
      linY = -touchX * 0.18; // Left/right (inverted for duck)
    }
    if (touchRotL) angZ = 0.5;
    if (touchRotR) angZ = -0.5;

    onnxController.setCommand(linX, linY, angZ);

    // Head control (joystick or keyboard 1/2)
    let neckPitch = onnxController.defaultNeckPitchCommand;
    let headYaw = 0;
    if (keys['Digit1']) neckPitch = 0.8;
    if (keys['Digit2']) neckPitch = -0.3;
    if (Math.abs(headY) > 0.1) neckPitch = headY * 0.8;
    if (Math.abs(headX) > 0.1) headYaw = -headX * 1.0;
    onnxController.commands[3] = neckPitch;
    onnxController.commands[5] = headYaw;
  }

  if (activeController === 'cpg' && cpgController && cpgController.enabled) {
    let fwd = 0;
    let lat = 0;
    let turn = 0;

    // Keyboard
    if (kbFwd) fwd = 0.8;
    if (kbBack) fwd = -0.4;
    if (kbLeft) lat = 0.3;
    if (kbRight) lat = -0.3;
    if (kbRotL) turn = 0.5;
    if (kbRotR) turn = -0.5;

    // Touch joystick override
    if (Math.abs(touchY) > 0.15 || Math.abs(touchX) > 0.15) {
      fwd = touchY * 0.8;
      lat = -touchX * 0.3;
    }
    if (touchRotL) turn = 0.5;
    if (touchRotR) turn = -0.5;

    cpgController.setCommand(fwd, lat, turn);
  }
}

window.addEventListener('keydown', (e) => {
  // Ignore if user is typing in an input/select
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

  keys[e.code] = true;

  if (e.code === 'Space') {
    paused = !paused;
    e.preventDefault();
  }
  if (e.code === 'KeyP') {
    toggleController();
  }
  if (e.code === 'KeyR') {
    resetScene();
  }
  if (e.code === 'KeyC') {
    cameraFollow = !cameraFollow;
  }
  if (e.code === 'KeyH') {
    if (helpOverlay) helpOverlay.style.display = helpOverlay.style.display === 'none' ? '' : 'none';
  }
  if (e.code === 'KeyF') {
    spawnObstacle(Math.random() < 0.5 ? 'ball' : 'box');
  }
});

window.addEventListener('keyup', (e) => {
  keys[e.code] = false;
});

// ─── UI Events ──────────────────────────────────────────────────────
sceneSelect.addEventListener('change', async (e) => {
  try {
    await loadScene(e.target.value);
  } catch (err) {
    setStatus(`Failed: ${e.target.value}`);
    console.error(err);
  }
});

resetBtn.addEventListener('click', resetScene);

if (controllerBtn) {
  controllerBtn.addEventListener('click', toggleController);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ─── Update Loop ────────────────────────────────────────────────────
function updateBodies() {
  for (const b in bodies) {
    const body = bodies[b];
    const idx = parseInt(b);
    getPosition(data.xpos, idx, body.position);
    getQuaternion(data.xquat, idx, body.quaternion);
    body.updateWorldMatrix(false, false);
  }
}

function followCamera() {
  if (!cameraFollow || !model || !data) return;

  // Follow the root body (index 1 = first non-world body)
  const rootBody = 1;
  const x = data.xpos[rootBody * 3 + 0];
  const y = data.xpos[rootBody * 3 + 1];
  const z = data.xpos[rootBody * 3 + 2];

  // Swizzle: MuJoCo (x,y,z) → Three (x,z,-y)
  controls.target.lerp(new THREE.Vector3(x, z, -y), 0.05);
}

// ─── Mobile Touch Controls ──────────────────────────────────────────
function setupTouch() {
  if (!isTouchDevice) return;

  const joystickZone = document.getElementById('joystick-zone');
  const joystickBase = document.getElementById('joystick-base');
  const joystickThumb = document.getElementById('joystick-thumb');
  const mobilePanel = document.getElementById('mobile-panel');
  const helpOverlayEl = document.getElementById('help-overlay');

  // Show mobile UI, hide desktop help
  if (joystickZone) joystickZone.style.display = 'block';
  if (mobilePanel) mobilePanel.style.display = 'flex';
  if (helpOverlayEl) helpOverlayEl.style.display = 'none';

  // Virtual joystick
  if (joystickBase && joystickThumb) {
    const baseRadius = 65; // half of 130px
    const thumbHalf = 24;  // half of 48px
    const maxDist = 40;

    let joystickActive = false;

    const updateThumb = (clientX, clientY) => {
      const rect = joystickBase.getBoundingClientRect();
      const cx = rect.left + baseRadius;
      const cy = rect.top + baseRadius;
      let dx = clientX - cx;
      let dy = clientY - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxDist) {
        dx = (dx / dist) * maxDist;
        dy = (dy / dist) * maxDist;
      }
      joystickThumb.style.left = (baseRadius - thumbHalf + dx) + 'px';
      joystickThumb.style.top = (baseRadius - thumbHalf + dy) + 'px';
      touchX = dx / maxDist;
      touchY = -dy / maxDist;
    };

    const resetThumb = () => {
      joystickThumb.style.left = (baseRadius - thumbHalf) + 'px';
      joystickThumb.style.top = (baseRadius - thumbHalf) + 'px';
      touchX = 0;
      touchY = 0;
      joystickActive = false;
    };

    joystickZone.addEventListener('touchstart', (e) => {
      e.preventDefault();
      joystickActive = true;
      updateThumb(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: false });

    joystickZone.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (!joystickActive) return;
      updateThumb(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: false });

    joystickZone.addEventListener('touchend', (e) => {
      e.preventDefault();
      resetThumb();
    }, { passive: false });

    joystickZone.addEventListener('touchcancel', resetThumb);
  }

  // Head control joystick (right panel, orange)
  const headJoystick = document.getElementById('head-joystick');
  const headJBase = document.getElementById('head-jbase');
  const headJThumb = document.getElementById('head-jthumb');

  if (headJoystick && headJBase && headJThumb) {
    const hR = 45, hT = 18, hMax = 30;
    let headTid = null;

    const updateHead = (x, y) => {
      const r = headJBase.getBoundingClientRect();
      let dx = x - (r.left + hR), dy = y - (r.top + hR);
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > hMax) { dx *= hMax / d; dy *= hMax / d; }
      headJThumb.style.left = (hR - hT + dx) + 'px';
      headJThumb.style.top = (hR - hT + dy) + 'px';
      headX = dx / hMax;
      headY = -dy / hMax;
    };

    const resetHead = () => {
      headJThumb.style.left = (hR - hT) + 'px';
      headJThumb.style.top = (hR - hT) + 'px';
      headX = 0; headY = 0; headTid = null;
    };

    const findTouch = (touches, id) => {
      for (let i = 0; i < touches.length; i++) {
        if (touches[i].identifier === id) return touches[i];
      }
      return null;
    };

    headJoystick.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const t = e.changedTouches[0];
      headTid = t.identifier;
      updateHead(t.clientX, t.clientY);
    }, { passive: false });

    headJoystick.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (headTid === null) return;
      const t = findTouch(e.touches, headTid);
      if (t) updateHead(t.clientX, t.clientY);
    }, { passive: false });

    headJoystick.addEventListener('touchend', (e) => {
      e.preventDefault();
      if (headTid !== null && findTouch(e.changedTouches, headTid)) resetHead();
    }, { passive: false });

    headJoystick.addEventListener('touchcancel', resetHead);
  }

  // All buttons in the right panel (unified: rotation + actions)
  if (mobilePanel) {
    mobilePanel.querySelectorAll('[data-action]').forEach(btn => {
      const action = btn.dataset.action;

      // Rotation: hold to rotate
      if (action === 'rotL' || action === 'rotR') {
        btn.addEventListener('touchstart', (e) => {
          e.preventDefault();
          if (action === 'rotL') touchRotL = true;
          if (action === 'rotR') touchRotR = true;
        }, { passive: false });
        btn.addEventListener('touchend', (e) => {
          e.preventDefault();
          if (action === 'rotL') touchRotL = false;
          if (action === 'rotR') touchRotR = false;
        }, { passive: false });
        btn.addEventListener('touchcancel', () => {
          touchRotL = false; touchRotR = false;
        });
      }

      // Tap actions
      if (action === 'ball') {
        btn.addEventListener('touchstart', (e) => { e.preventDefault(); spawnObstacle('ball'); }, { passive: false });
      }
      if (action === 'box') {
        btn.addEventListener('touchstart', (e) => { e.preventDefault(); spawnObstacle('box'); }, { passive: false });
      }
      if (action === 'toggle') {
        btn.addEventListener('touchstart', (e) => { e.preventDefault(); toggleController(); }, { passive: false });
      }
    });
  }
}

// ─── Boot ───────────────────────────────────────────────────────────
(async () => {
  try {
    setStatus('Loading MuJoCo WASM...');
    mujoco = await load_mujoco();

    // Ensure /working dir exists
    if (!mujoco.FS.analyzePath('/working').exists) {
      mujoco.FS.mkdir('/working');
    }

    // Default scene: OpenDuck Backlash (best ONNX walking)
    await loadScene('openduck/scene_flat_terrain_backlash.xml');
    setupTouch();
  } catch (e) {
    setStatus('Boot failed');
    console.error(e);
    return;
  }

  // Physics substep budget per render frame.
  // 500Hz physics / 60fps = ~8 substeps. Cap at 20 for safety.
  const MAX_SUBSTEPS = 20;

  async function animate() {
    if (model && data && !paused) {
      handleInput();

      const timestep = model.opt.timestep;
      const frameDt = 1.0 / 60.0;
      const nsteps = Math.min(Math.round(frameDt / timestep), MAX_SUBSTEPS);

      for (let s = 0; s < nsteps; s++) {
        // CPG: compute & apply torques BEFORE physics integration
        if (activeController === 'cpg' && cpgController) {
          cpgController.step();
        }

        // Physics step
        mujoco.mj_step(model, data);
        stepCounter++;

        // ONNX: run policy AFTER physics step at decimation boundary.
        // Use await so ctrl is applied before next substep batch.
        if (activeController === 'onnx' && onnxController && onnxController.enabled) {
          onnxController.stepCounter = stepCounter;
          if (stepCounter % onnxController.decimation === 0) {
            await onnxController.runPolicy();
          }
        }
      }

      updateBodies();
      followCamera();
    }

    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
