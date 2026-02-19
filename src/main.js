/**
 * main.js — MuJoCo WASM + Three.js playground with walking controllers.
 * Supports: humanoid (physics only), OpenDuck (ONNX), Unitree H1 (CPG).
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
scene.add(dirLight);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(2.0, 1.6, 2.0);
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

// Step counter for ONNX decimation
let stepCounter = 0;

// ─── Scene Config ───────────────────────────────────────────────────
const SCENES = {
  'humanoid.xml': {
    controller: null,
    camera: { pos: [2.0, 1.6, 2.0], target: [0, 0.9, 0] },
  },
  'openduck/scene_flat_terrain.xml': {
    controller: 'onnx',
    camera: { pos: [0.5, 0.4, 0.5], target: [0, 0.15, 0] },
  },
  'unitree_h1/scene.xml': {
    controller: 'cpg',
    camera: { pos: [3.0, 2.0, 3.0], target: [0, 0.9, 0] },
  },
};

let currentScenePath = 'humanoid.xml';

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

  // Clean up old model
  clearScene();
  if (data) { data.delete(); data = null; }
  if (model) { model.delete(); model = null; }

  // Load MuJoCo model
  model = mujoco.MjModel.loadFromXML('/working/' + scenePath);
  data = new mujoco.MjData(model);

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
    onnxController = new OnnxController(mujoco, model, data);
    const loaded = await onnxController.loadModel('./assets/models/openduck_walk.onnx');
    if (loaded) {
      // Warm up physics
      for (let i = 0; i < 100; i++) mujoco.mj_step(model, data);
      onnxController.enabled = true;
      activeController = 'onnx';
      updateControllerBtn();
    }
  } else if (cfg.controller === 'cpg') {
    // Unitree H1 CPG walking
    cpgController = new CpgController(mujoco, model, data);
    // Warm up physics
    for (let i = 0; i < 100; i++) mujoco.mj_step(model, data);
    cpgController.enabled = true;
    activeController = 'cpg';
    updateControllerBtn();
  }

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

  // Warm up physics
  for (let i = 0; i < 100; i++) mujoco.mj_step(model, data);

  stepCounter = 0;

  if (onnxController) {
    onnxController.reset();
    onnxController.enabled = true;
  }
  if (cpgController) {
    cpgController.reset();
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
function handleKeyboard() {
  if (activeController === 'onnx' && onnxController && onnxController.enabled) {
    let linX = onnxController.defaultForwardCommand;
    let linY = 0;
    let angZ = 0;

    if (keys['KeyW'] || keys['ArrowUp']) linX = 0.10;
    if (keys['KeyS'] || keys['ArrowDown']) linX = -0.10;
    if (keys['KeyA'] || keys['ArrowLeft']) linY = 0.15;
    if (keys['KeyD'] || keys['ArrowRight']) linY = -0.15;
    if (keys['KeyQ']) angZ = 0.5;
    if (keys['KeyE']) angZ = -0.5;

    onnxController.setCommand(linX, linY, angZ);
  }

  if (activeController === 'cpg' && cpgController && cpgController.enabled) {
    let fwd = 0;
    let lat = 0;
    let turn = 0;

    if (keys['KeyW'] || keys['ArrowUp']) fwd = 0.8;
    if (keys['KeyS'] || keys['ArrowDown']) fwd = -0.4;
    if (keys['KeyA'] || keys['ArrowLeft']) lat = 0.3;
    if (keys['KeyD'] || keys['ArrowRight']) lat = -0.3;
    if (keys['KeyQ']) turn = 0.5;
    if (keys['KeyE']) turn = -0.5;

    cpgController.setCommand(fwd, lat, turn);
  }
}

window.addEventListener('keydown', (e) => {
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
  let rootBody = 1;
  const x = data.xpos[rootBody * 3 + 0];
  const y = data.xpos[rootBody * 3 + 1];
  const z = data.xpos[rootBody * 3 + 2];

  // Swizzle: MuJoCo (x,y,z) → Three (x,z,-y)
  const tx = x;
  const ty = z;
  const tz = -y;

  // Smooth follow
  controls.target.lerp(new THREE.Vector3(tx, ty, tz), 0.05);
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

    await loadScene('humanoid.xml');
    updateControllerBtn();
  } catch (e) {
    setStatus('Boot failed');
    console.error(e);
    return;
  }

  // Real-time physics: run enough substeps per frame to match wall-clock time.
  // MuJoCo timestep: 0.002s (500Hz) for OpenDuck/H1, 0.005s (200Hz) for humanoid.
  const MAX_SUBSTEPS = 20; // Safety cap to prevent freeze on slow frames

  function animate() {
    requestAnimationFrame(animate);

    if (model && data && !paused) {
      handleKeyboard();

      const timestep = model.opt.timestep;
      const frameDt = 1.0 / 60.0; // Target 60fps real-time
      const nsteps = Math.min(Math.round(frameDt / timestep), MAX_SUBSTEPS);

      for (let s = 0; s < nsteps; s++) {
        // CPG: set torques BEFORE physics integration
        if (activeController === 'cpg' && cpgController) {
          cpgController.step();
        }

        // Physics step
        mujoco.mj_step(model, data);
        stepCounter++;

        // ONNX: run policy AFTER physics step at decimation boundary
        if (activeController === 'onnx' && onnxController && onnxController.enabled) {
          onnxController.stepCounter = stepCounter;
          if (stepCounter % onnxController.decimation === 0) {
            onnxController.runPolicyAsync();
          }
        }
      }

      updateBodies();
      followCamera();
    }

    controls.update();
    renderer.render(scene, camera);
  }

  animate();
})();
