/**
 * meshBuilder.js — Build Three.js geometries/materials from MuJoCo model data.
 * Handles mesh (type 7/mjGEOM_MESH), plane (type 0), and RGBA colors/textures.
 * Adapted from mujoco-web/src/mujocoUtils.js.
 */

import * as THREE from 'three';

/**
 * Coordinate swizzle helpers: MuJoCo (x,y,z) → Three.js (x,z,-y)
 */
export function getPosition(buffer, index, target) {
  return target.set(
    buffer[(index * 3) + 0],
    buffer[(index * 3) + 2],
   -buffer[(index * 3) + 1]);
}

export function getQuaternion(buffer, index, target) {
  return target.set(
   -buffer[(index * 4) + 1],
   -buffer[(index * 4) + 3],
    buffer[(index * 4) + 2],
   -buffer[(index * 4) + 0]);
}

/**
 * Build all Three.js bodies for a MuJoCo scene.
 * Returns { mujocoRoot, bodies } where bodies is indexed by body ID.
 */
export function buildScene(model) {
  const textDecoder = new TextDecoder('utf-8');
  const namesArray = new Uint8Array(model.names);

  const mujocoRoot = new THREE.Group();
  mujocoRoot.name = 'MuJoCo Root';

  const bodies = {};
  const meshCache = {};

  for (let g = 0; g < model.ngeom; g++) {
    // Only visualize geom groups 0-2 (default behavior)
    if (!(model.geom_group[g] < 3)) continue;

    const b = model.geom_bodyid[g];
    const type = model.geom_type[g];
    const size = [
      model.geom_size[(g * 3) + 0],
      model.geom_size[(g * 3) + 1],
      model.geom_size[(g * 3) + 2]
    ];

    // Create body group if needed
    if (!(b in bodies)) {
      bodies[b] = new THREE.Group();
      const start = model.name_bodyadr[b];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) end++;
      bodies[b].name = textDecoder.decode(namesArray.subarray(start, end));
      bodies[b].bodyID = b;
    }

    // Build geometry based on type
    let geometry;

    if (type === 0) {
      // PLANE — flat ground
      geometry = new THREE.PlaneGeometry(100, 100);
    } else if (type === 2) {
      // SPHERE
      geometry = new THREE.SphereGeometry(size[0], 20, 16);
    } else if (type === 3) {
      // CAPSULE
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 12, 16);
    } else if (type === 4) {
      // ELLIPSOID — stretched sphere
      geometry = new THREE.SphereGeometry(1, 20, 16);
    } else if (type === 5) {
      // CYLINDER
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0, 20);
    } else if (type === 6) {
      // BOX — note MuJoCo→Three swizzle: (x,z,y) sizes
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
    } else if (type === 7) {
      // MESH — read from model buffers with caching
      const meshID = model.geom_dataid[g];
      if (meshID in meshCache) {
        geometry = meshCache[meshID];
      } else {
        geometry = buildMeshGeometry(model, meshID);
        meshCache[meshID] = geometry;
      }
    } else {
      // Fallback sphere
      geometry = new THREE.SphereGeometry(Math.max(0.01, size[0] || 0.03), 10, 8);
    }

    // Build material with RGBA colors
    const material = buildMaterial(model, g);

    let mesh;
    if (type === 0) {
      // Ground plane
      mesh = new THREE.Mesh(geometry, material);
      mesh.rotateX(-Math.PI / 2);
      mesh.receiveShadow = true;
      mesh.castShadow = false;
    } else {
      mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }

    mesh.bodyID = b;
    bodies[b].add(mesh);

    // Set local position/rotation within body
    getPosition(model.geom_pos, g, mesh.position);
    if (type !== 0) {
      getQuaternion(model.geom_quat, g, mesh.quaternion);
    }
    if (type === 4) {
      mesh.scale.set(size[0], size[2], size[1]);
    }
  }

  // Assemble body hierarchy: all bodies are children of body 0 (world)
  for (let b = 0; b < model.nbody; b++) {
    if (!bodies[b]) {
      bodies[b] = new THREE.Group();
      bodies[b].bodyID = b;
      bodies[b].name = `body_${b}`;
    }
    if (b === 0) {
      mujocoRoot.add(bodies[b]);
    } else {
      (bodies[0] || mujocoRoot).add(bodies[b]);
    }
  }

  return { mujocoRoot, bodies };
}

/**
 * Build a THREE.BufferGeometry from MuJoCo mesh data.
 * Swizzles vertex/normal coordinates from MuJoCo→Three.js handedness.
 */
function buildMeshGeometry(model, meshID) {
  const geometry = new THREE.BufferGeometry();

  // Extract vertex positions
  const vertStart = model.mesh_vertadr[meshID] * 3;
  const vertCount = model.mesh_vertnum[meshID] * 3;
  const vertexBuffer = new Float32Array(model.mesh_vert.subarray(vertStart, vertStart + vertCount));

  // Swizzle vertices: (x,y,z) → (x,z,-y)
  for (let v = 0; v < vertexBuffer.length; v += 3) {
    const temp = vertexBuffer[v + 1];
    vertexBuffer[v + 1] = vertexBuffer[v + 2];
    vertexBuffer[v + 2] = -temp;
  }

  // Extract normals
  let normalAdr, normalNum;
  if (model.mesh_normaladr) {
    normalAdr = model.mesh_normaladr[meshID];
    normalNum = model.mesh_normalnum[meshID];
  } else {
    normalAdr = model.mesh_vertadr[meshID];
    normalNum = model.mesh_vertnum[meshID];
  }
  const normalBuffer = new Float32Array(model.mesh_normal.subarray(normalAdr * 3, (normalAdr + normalNum) * 3));

  // Swizzle normals
  for (let v = 0; v < normalBuffer.length; v += 3) {
    const temp = normalBuffer[v + 1];
    normalBuffer[v + 1] = normalBuffer[v + 2];
    normalBuffer[v + 2] = -temp;
  }

  // Face indices
  const faceStart = model.mesh_faceadr[meshID] * 3;
  const faceCount = model.mesh_facenum[meshID] * 3;
  const faceBuffer = model.mesh_face.subarray(faceStart, faceStart + faceCount);

  // Swizzle UV and normals to per-vertex format (MuJoCo uses per-face indexing)
  const numVerts = model.mesh_vertnum[meshID];
  const swizzledNormals = new Float32Array(numVerts * 3);

  if (model.mesh_facenormal) {
    const faceNormalBuffer = model.mesh_facenormal.subarray(faceStart, faceStart + faceCount);
    for (let t = 0; t < faceCount / 3; t++) {
      const vi0 = faceBuffer[t * 3 + 0];
      const vi1 = faceBuffer[t * 3 + 1];
      const vi2 = faceBuffer[t * 3 + 2];
      const ni0 = faceNormalBuffer[t * 3 + 0];
      const ni1 = faceNormalBuffer[t * 3 + 1];
      const ni2 = faceNormalBuffer[t * 3 + 2];

      for (let c = 0; c < 3; c++) {
        swizzledNormals[vi0 * 3 + c] = normalBuffer[ni0 * 3 + c];
        swizzledNormals[vi1 * 3 + c] = normalBuffer[ni1 * 3 + c];
        swizzledNormals[vi2 * 3 + c] = normalBuffer[ni2 * 3 + c];
      }
    }
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(vertexBuffer, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(swizzledNormals, 3));
  geometry.setIndex(Array.from(faceBuffer));
  geometry.computeVertexNormals();

  return geometry;
}

/**
 * Build a MeshPhysicalMaterial from geom/material RGBA colors.
 */
function buildMaterial(model, g) {
  let color = [
    model.geom_rgba[(g * 4) + 0],
    model.geom_rgba[(g * 4) + 1],
    model.geom_rgba[(g * 4) + 2],
    model.geom_rgba[(g * 4) + 3]
  ];

  let texture = null;

  if (model.geom_matid[g] !== -1) {
    const matId = model.geom_matid[g];
    color = [
      model.mat_rgba[(matId * 4) + 0],
      model.mat_rgba[(matId * 4) + 1],
      model.mat_rgba[(matId * 4) + 2],
      model.mat_rgba[(matId * 4) + 3]
    ];

    // Try to load texture from material
    if (model.mat_texid) {
      // mat_texid is a matrix (nmat x mjNTEXROLE) in newer MuJoCo
      const mjNTEXROLE = 10;
      const mjTEXROLE_RGB = 1;
      let texId = -1;
      try {
        texId = model.mat_texid[(matId * mjNTEXROLE) + mjTEXROLE_RGB];
      } catch (e) {
        // Older format: simple array
        try { texId = model.mat_texid[matId]; } catch (e2) { /* no texture */ }
      }

      if (texId !== undefined && texId !== -1 && model.tex_data) {
        try {
          const width = model.tex_width[texId];
          const height = model.tex_height[texId];
          const offset = model.tex_adr[texId];
          const channels = model.tex_nchannel ? model.tex_nchannel[texId] : 3;
          const texData = model.tex_data;
          const rgbaArray = new Uint8Array(width * height * 4);
          for (let p = 0; p < width * height; p++) {
            rgbaArray[(p * 4) + 0] = texData[offset + (p * channels) + 0];
            rgbaArray[(p * 4) + 1] = channels > 1 ? texData[offset + (p * channels) + 1] : rgbaArray[(p * 4)];
            rgbaArray[(p * 4) + 2] = channels > 2 ? texData[offset + (p * channels) + 2] : rgbaArray[(p * 4)];
            rgbaArray[(p * 4) + 3] = channels > 3 ? texData[offset + (p * channels) + 3] : 255;
          }
          texture = new THREE.DataTexture(rgbaArray, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
          if (model.mat_texrepeat) {
            texture.repeat.set(
              model.mat_texrepeat[(matId * 2) + 0],
              model.mat_texrepeat[(matId * 2) + 1]
            );
          }
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
          texture.needsUpdate = true;
        } catch (e) {
          // Texture loading failed, continue without
        }
      }
    }
  }

  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(color[0], color[1], color[2]),
    transparent: color[3] < 1.0,
    opacity: color[3],
    roughness: 0.7,
    metalness: 0.1,
    map: texture
  });
}
