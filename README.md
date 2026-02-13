# mujoco-js playground

Browser-first MuJoCo playground based on `mujoco-js`.

## Goal
- Keep a simple, stable baseline viewer in the browser.
- Add Unitree scenes/models incrementally under `assets/scenes/unitree/`.

## Run
1. Install deps:
   `npm install`
2. Build bundle:
   `npm run build`
3. Start local server:
   `npm run dev`
4. Open:
   `http://localhost:8080`

## Scenes
- `assets/scenes/humanoid.xml` (working baseline)
- `assets/scenes/unitree/` (placeholder for Unitree XML + assets)

## Notes
- Current renderer is intentionally minimal (collision/debug-style geometry).
- Next step: import a Unitree XML (Go2/H1) and resolve asset paths for web.
