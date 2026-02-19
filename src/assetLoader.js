/**
 * assetLoader.js — Fetch binary/text assets and write to MuJoCo VFS.
 * Adapted from mujoco-web/src/mujocoUtils.js downloadExampleScenesFolder.
 */

/** Asset manifests per scene */
const SCENE_ASSETS = {
  'humanoid.xml': ['humanoid.xml'],

  'openduck/scene_flat_terrain.xml': [
    'openduck/scene_flat_terrain.xml',
    'openduck/open_duck_mini_v2.xml',
    'openduck/sensors.xml',
    'openduck/joints_properties.xml',
    'openduck/assets/antenna.stl',
    'openduck/assets/body_back.stl',
    'openduck/assets/body_front.stl',
    'openduck/assets/body_middle_bottom.stl',
    'openduck/assets/body_middle_top.stl',
    'openduck/assets/foot_bottom_pla.stl',
    'openduck/assets/foot_bottom_tpu.stl',
    'openduck/assets/foot_side.stl',
    'openduck/assets/foot_top.stl',
    'openduck/assets/head_bot_sheet.stl',
    'openduck/assets/head_pitch_to_yaw.stl',
    'openduck/assets/head.stl',
    'openduck/assets/head_yaw_to_roll.stl',
    'openduck/assets/hfield.png',
    'openduck/assets/left_antenna_holder.stl',
    'openduck/assets/left_cache.stl',
    'openduck/assets/left_knee_to_ankle_left_sheet.stl',
    'openduck/assets/left_knee_to_ankle_right_sheet.stl',
    'openduck/assets/left_roll_to_pitch.stl',
    'openduck/assets/leg_spacer.stl',
    'openduck/assets/neck_left_sheet.stl',
    'openduck/assets/neck_right_sheet.stl',
    'openduck/assets/right_antenna_holder.stl',
    'openduck/assets/right_cache.stl',
    'openduck/assets/right_roll_to_pitch.stl',
    'openduck/assets/roll_motor_bottom.stl',
    'openduck/assets/roll_motor_top.stl',
    'openduck/assets/trunk_bottom.stl',
    'openduck/assets/trunk_top.stl',
  ],

  'openduck/scene_flat_terrain_backlash.xml': [
    'openduck/scene_flat_terrain_backlash.xml',
    'openduck/open_duck_mini_v2_backlash.xml',
    'openduck/sensors.xml',
    'openduck/joints_properties.xml',
    'openduck/assets/antenna.stl',
    'openduck/assets/body_back.stl',
    'openduck/assets/body_front.stl',
    'openduck/assets/body_middle_bottom.stl',
    'openduck/assets/body_middle_top.stl',
    'openduck/assets/foot_bottom_pla.stl',
    'openduck/assets/foot_bottom_tpu.stl',
    'openduck/assets/foot_side.stl',
    'openduck/assets/foot_top.stl',
    'openduck/assets/head_bot_sheet.stl',
    'openduck/assets/head_pitch_to_yaw.stl',
    'openduck/assets/head.stl',
    'openduck/assets/head_yaw_to_roll.stl',
    'openduck/assets/hfield.png',
    'openduck/assets/left_antenna_holder.stl',
    'openduck/assets/left_cache.stl',
    'openduck/assets/left_knee_to_ankle_left_sheet.stl',
    'openduck/assets/left_knee_to_ankle_right_sheet.stl',
    'openduck/assets/left_roll_to_pitch.stl',
    'openduck/assets/leg_spacer.stl',
    'openduck/assets/neck_left_sheet.stl',
    'openduck/assets/neck_right_sheet.stl',
    'openduck/assets/right_antenna_holder.stl',
    'openduck/assets/right_cache.stl',
    'openduck/assets/right_roll_to_pitch.stl',
    'openduck/assets/roll_motor_bottom.stl',
    'openduck/assets/roll_motor_top.stl',
    'openduck/assets/trunk_bottom.stl',
    'openduck/assets/trunk_top.stl',
  ],

  'unitree_h1/scene.xml': [
    'unitree_h1/scene.xml',
    'unitree_h1/h1.xml',
    'unitree_h1/assets/pelvis.stl',
    'unitree_h1/assets/left_hip_yaw_link.stl',
    'unitree_h1/assets/left_hip_roll_link.stl',
    'unitree_h1/assets/left_hip_pitch_link.stl',
    'unitree_h1/assets/left_knee_link.stl',
    'unitree_h1/assets/left_ankle_link.stl',
    'unitree_h1/assets/right_hip_yaw_link.stl',
    'unitree_h1/assets/right_hip_roll_link.stl',
    'unitree_h1/assets/right_hip_pitch_link.stl',
    'unitree_h1/assets/right_knee_link.stl',
    'unitree_h1/assets/right_ankle_link.stl',
    'unitree_h1/assets/torso_link.stl',
    'unitree_h1/assets/left_shoulder_pitch_link.stl',
    'unitree_h1/assets/left_shoulder_roll_link.stl',
    'unitree_h1/assets/left_shoulder_yaw_link.stl',
    'unitree_h1/assets/left_elbow_link.stl',
    'unitree_h1/assets/right_shoulder_pitch_link.stl',
    'unitree_h1/assets/right_shoulder_roll_link.stl',
    'unitree_h1/assets/right_shoulder_yaw_link.stl',
    'unitree_h1/assets/right_elbow_link.stl',
    'unitree_h1/assets/logo_link.stl',
  ],
};

/** Track which files have already been loaded to VFS */
const loadedFiles = new Set();

/**
 * Load all assets for a scene into MuJoCo's virtual filesystem.
 * Skips files already loaded. Returns when all files are written.
 */
export async function loadSceneAssets(mujoco, scenePath, onProgress) {
  const fileList = SCENE_ASSETS[scenePath];
  if (!fileList) {
    console.warn(`No asset manifest for "${scenePath}", loading single file.`);
    await loadSingleFile(mujoco, scenePath);
    return;
  }

  // Filter to only new files
  const toLoad = fileList.filter(f => !loadedFiles.has(f));
  if (toLoad.length === 0) return;

  if (onProgress) onProgress(`Fetching ${toLoad.length} assets...`);

  // Fetch all in parallel
  const cacheBust = '?v=' + Date.now();
  const responses = await Promise.all(
    toLoad.map(url => fetch('./assets/scenes/' + url + cacheBust))
  );

  // Write to VFS
  for (let i = 0; i < toLoad.length; i++) {
    const file = toLoad[i];
    const resp = responses[i];
    if (!resp.ok) {
      console.warn(`Failed to fetch ${file}: ${resp.status}`);
      continue;
    }

    // Ensure directory structure exists
    ensureDir(mujoco, '/working/' + file);

    // Write binary or text
    if (file.endsWith('.stl') || file.endsWith('.png') || file.endsWith('.obj')) {
      mujoco.FS.writeFile('/working/' + file, new Uint8Array(await resp.arrayBuffer()));
    } else {
      mujoco.FS.writeFile('/working/' + file, await resp.text());
    }

    loadedFiles.add(file);
  }
}

/**
 * Load a single XML file (for simple scenes like humanoid.xml).
 */
async function loadSingleFile(mujoco, scenePath) {
  if (loadedFiles.has(scenePath)) return;
  const resp = await fetch('./assets/scenes/' + scenePath);
  if (!resp.ok) throw new Error(`Scene not found: ${scenePath}`);
  ensureDir(mujoco, '/working/' + scenePath);
  mujoco.FS.writeFile('/working/' + scenePath, await resp.text());
  loadedFiles.add(scenePath);
}

/**
 * Ensure all directories in a VFS path exist.
 */
function ensureDir(mujoco, fullPath) {
  const dir = fullPath.substring(0, fullPath.lastIndexOf('/'));
  const parts = dir.split('/').filter(Boolean);
  let cur = '';
  for (const p of parts) {
    cur += '/' + p;
    if (!mujoco.FS.analyzePath(cur).exists) {
      mujoco.FS.mkdir(cur);
    }
  }
}
