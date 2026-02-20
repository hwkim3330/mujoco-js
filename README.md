# mujoco-js playground

Browser-based MuJoCo robot simulation with ONNX neural-network walking policy.

**Live demo:** https://hwkim3330.github.io/mujoco-js/

## Features
- MuJoCo WASM physics (mujoco-js 0.0.7)
- ONNX Runtime Web for OpenDuck Mini RL walking policy
- Three.js 3D rendering with shadows and camera follow
- Mobile touch controls (movement joystick + head joystick)
- Obstacle spawning (balls & boxes)
- Simulation speed control (0.25x ~ 4x)

## Scenes
| Scene | Controller | Description |
|-------|-----------|-------------|
| OpenDuck Mini (ONNX) | Neural network | Best walking, with backlash joints |
| OpenDuck (no backlash) | Neural network | Simpler joint model |
| Unitree H1 (CPG) | Procedural CPG | Humanoid walking |

## Controls

### Desktop (Keyboard)
| Key | Action |
|-----|--------|
| `W` `A` `S` `D` | Move |
| `Q` `E` | Rotate |
| `1` `2` | Head up / down |
| `P` | Toggle controller |
| `R` | Reset pose |
| `F` | Spawn obstacle |
| `[` `]` | Speed down / up |
| `Space` | Pause |
| `C` | Camera follow |
| `H` | Toggle help |

### Mobile (Touch)
- **Left joystick**: Movement (forward/back/strafe)
- **Right orange joystick (HEAD)**: Head pitch & yaw
- **Rotation buttons**: Turn left/right
- **Spawn buttons**: Ball / Box

### All Devices
- **Speed button** (top bar): Cycle through 0.25x / 0.5x / 1x / 2x / 4x
- **Head joystick**: Also works with mouse on desktop

## Run Locally
```bash
npm install
npm run build
npm run dev
# Open http://localhost:8080
```

## Tech Stack
- [mujoco-js](https://github.com/nicholasgasior/mujoco-js) 0.0.7 — MuJoCo WASM bindings
- [ONNX Runtime Web](https://onnxruntime.ai/) 1.17.0 — Neural network inference
- [Three.js](https://threejs.org/) 0.181.0 — 3D rendering
- esbuild — Bundler
