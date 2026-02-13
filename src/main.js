import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import load_mujoco from 'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js';

const statusEl = document.getElementById('status');
const sceneSelect = document.getElementById('scene-select');
const resetBtn = document.getElementById('btn-reset');

const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11151d);
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.0));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(3, 5, 3);
scene.add(dir);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(2.0, 1.6, 2.0);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;

let mujoco;
let model;
let data;
let geomMeshes = [];

function setStatus(text) {
  statusEl.textContent = text;
}

function clearGeomMeshes() {
  for (const m of geomMeshes) scene.remove(m);
  geomMeshes = [];
}

function makeGeomMesh(geomType, size) {
  let geometry;
  switch (geomType) {
    case 2: // sphere
      geometry = new THREE.SphereGeometry(size[0], 16, 12);
      break;
    case 3: // capsule (approx with cylinder)
      geometry = new THREE.CapsuleGeometry(size[0], Math.max(0.001, size[1] * 2.0), 4, 8);
      break;
    case 5: // cylinder
      geometry = new THREE.CylinderGeometry(size[0], size[0], Math.max(0.001, size[1] * 2.0), 12);
      break;
    case 6: // box
      geometry = new THREE.BoxGeometry(Math.max(0.001, size[0] * 2), Math.max(0.001, size[1] * 2), Math.max(0.001, size[2] * 2));
      break;
    default:
      geometry = new THREE.SphereGeometry(Math.max(0.01, size[0] || 0.03), 10, 8);
      break;
  }
  const material = new THREE.MeshStandardMaterial({ color: 0x6ea8ff, roughness: 0.8, metalness: 0.1 });
  return new THREE.Mesh(geometry, material);
}

function rebuildSceneMeshes() {
  clearGeomMeshes();
  for (let g = 0; g < model.ngeom; g++) {
    const adr = g * 3;
    const size = [model.geom_size[adr + 0], model.geom_size[adr + 1], model.geom_size[adr + 2]];
    const mesh = makeGeomMesh(model.geom_type[g], size);
    scene.add(mesh);
    geomMeshes.push(mesh);
  }
}

function updateGeomMeshes() {
  for (let g = 0; g < model.ngeom; g++) {
    const mesh = geomMeshes[g];
    const p = g * 3;
    mesh.position.set(data.geom_xpos[p + 0], data.geom_xpos[p + 1], data.geom_xpos[p + 2]);

    const r = g * 9;
    const m = new THREE.Matrix4();
    m.set(
      data.geom_xmat[r + 0], data.geom_xmat[r + 1], data.geom_xmat[r + 2], 0,
      data.geom_xmat[r + 3], data.geom_xmat[r + 4], data.geom_xmat[r + 5], 0,
      data.geom_xmat[r + 6], data.geom_xmat[r + 7], data.geom_xmat[r + 8], 0,
      0, 0, 0, 1
    );
    mesh.quaternion.setFromRotationMatrix(m);
  }
}

async function loadSceneXML(scenePath) {
  if (!mujoco.FS.analyzePath('/working').exists) mujoco.FS.mkdir('/working');

  let xml;
  try {
    xml = await (await fetch(`./assets/scenes/${scenePath}`)).text();
  } catch (e) {
    throw new Error(`scene not found: ${scenePath}`);
  }

  const virtualPath = `/working/${scenePath}`;
  const dir = virtualPath.substring(0, virtualPath.lastIndexOf('/'));
  const parts = dir.split('/').filter(Boolean);
  let cur = '';
  for (const p of parts) {
    cur += `/${p}`;
    if (!mujoco.FS.analyzePath(cur).exists) mujoco.FS.mkdir(cur);
  }
  mujoco.FS.writeFile(virtualPath, xml);

  if (data) data.delete();
  if (model) model.delete();

  model = mujoco.MjModel.loadFromXML(virtualPath);
  data = new mujoco.MjData(model);

  if (model.nkey > 0) {
    data.qpos.set(model.key_qpos.slice(0, model.nq));
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    if (model.key_ctrl) data.ctrl.set(model.key_ctrl.slice(0, model.nu));
    mujoco.mj_forward(model, data);
  }

  rebuildSceneMeshes();
  controls.target.set(0, 0.9, 0);
  controls.update();
}

async function boot() {
  setStatus('Loading MuJoCo wasm...');
  mujoco = await load_mujoco();
  setStatus('Loading scene: humanoid.xml');
  await loadSceneXML('humanoid.xml');
  setStatus('Ready');
}

sceneSelect.addEventListener('change', async (e) => {
  const selected = e.target.value;
  try {
    setStatus(`Loading scene: ${selected}`);
    await loadSceneXML(selected);
    setStatus(`Ready: ${selected}`);
  } catch (err) {
    setStatus(`Failed: ${selected}`);
    console.warn(err);
  }
});

resetBtn.addEventListener('click', () => {
  if (!model || !data || model.nkey <= 0) return;
  data.qpos.set(model.key_qpos.slice(0, model.nq));
  for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
  if (model.key_ctrl) data.ctrl.set(model.key_ctrl.slice(0, model.nu));
  mujoco.mj_forward(model, data);
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

(async () => {
  try {
    await boot();
  } catch (e) {
    setStatus('Boot failed');
    console.error(e);
  }

  function animate() {
    requestAnimationFrame(animate);
    if (model && data) {
      mujoco.mj_step(model, data);
      updateGeomMeshes();
    }
    controls.update();
    renderer.render(scene, camera);
  }

  animate();
})();
