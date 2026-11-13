# footage2robot

Turn videos of people doing things into training data for robots.

Upload footage of a person performing a task. Each clip becomes one demonstration
episode: the person's body and hands are tracked in 3D, retargeted to robot states and
actions, and exported in the dataset formats robot-learning code already reads.

```
video ──► tracking ──► clean-up ──► retargeting ──► episodes ──► exporters
          (body + 2     (gap fill,   (5 robot        (state,      (6 formats)
           hands, 3D)    smoothing)   embodiments)    action,
                                                      images, task)
```

## What it produces

**Embodiments** — one dataset per robot type:

| Key | State | Notes |
| --- | --- | --- |
| `single_arm` | 7-D: end-effector xyz, rotation, gripper | most active hand, or pick left/right |
| `bimanual` | 14-D: both arms | |
| `dex_hand` | 25-D: wrist pose + 19 finger joint angles | 15 flexion + 4 spread |
| `humanoid` | 25-D: waist, head, arms, legs, hand closure | generic joint tree, radians |
| `keypoints` | 227-D: raw 3D body + hand keypoints | for representation learning / custom retargeting |

Poses are expressed in a body-centric frame (x forward, y left, z up, origin between the
shoulders), so the result does not depend on where the camera was. Gripper is `0` open → `1` closed.
Actions are either the next state (`absolute`) or the change to it (`delta`).

**Formats**:

| Key | Format | Used by |
| --- | --- | --- |
| `lerobot` | LeRobot v2.1 (Parquet + MP4) | Hugging Face LeRobot, ACT, Diffusion Policy, pi0, SmolVLA |
| `rlds` | RLDS / TFDS (TFRecord) | Open X-Embodiment, Octo, OpenVLA |
| `robomimic` | robomimic HDF5 | robomimic, RoboSuite, MimicGen |
| `zarr` | Diffusion Policy replay buffer | Diffusion Policy, UMI |
| `npz` | NumPy per episode + JSON | anything |
| `json` | Plain JSON, no images | inspection, web tools |

## Run it

Requirements: Python 3.10+, Node 18+, `ffmpeg`, and on headless Linux the EGL libraries
MediaPipe needs (`apt-get install libegl1 libgles2`).

```bash
# backend
cd backend
pip install -e ".[dev]"          # add ",rlds" for the RLDS exporter (TensorFlow)

# frontend (built once, then served by the backend)
cd ../frontend
npm install && npm run build

# web app at http://127.0.0.1:8000
cd ../backend
footage2robot serve
```

For frontend development run `npm run dev` (port 5173, proxies `/api` to port 8000).

### Command line

```bash
footage2robot convert demo1.mp4 demo2.mp4 -o out \
  --task "pick up the cup and place it on the shelf" \
  --embodiments single_arm,bimanual --formats lerobot,rlds

footage2robot formats            # list embodiments and formats
```

Output:

```
out/summary.json
out/motion/episode_000.npz                  tracked human motion
out/datasets/<embodiment>/<format>/...      exported datasets
```

### API

| Method | Path | |
| --- | --- | --- |
| `GET` | `/api/options` | embodiments, formats, limits |
| `POST` | `/api/jobs` | multipart `files[]` + `config` (JSON) → job |
| `POST` | `/api/jobs/demo` | run on generated motion, no footage needed |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | status, progress, summary |
| `GET` | `/api/jobs/{id}/clips/{n}/overlay` | 2D keypoints for the skeleton overlay |
| `GET` | `/api/jobs/{id}/datasets/{embodiment}/preview` | state trajectories |
| `GET` | `/api/jobs/{id}/download?embodiment=&format=` | zip of one export, or everything |
| `DELETE` | `/api/jobs/{id}` | |

Settings: `FOOTAGE2ROBOT_DATA_DIR` (job storage, default `./f2r_data`),
`FOOTAGE2ROBOT_MAX_UPLOAD_MB` (default 2048), `FOOTAGE2ROBOT_MODEL_DIR` (model cache).

## Tests

```bash
cd backend && pytest
```

The suite runs the whole pipeline on generated motion, loads every export back
(RLDS through `tfds.builder_from_directory`), and exercises the API.

## Layout

```
backend/footage2robot/
  extract/     video → HumanMotion (MediaPipe pose + hands; synthetic generator for tests)
  process/     gap filling, smoothing
  retarget/    HumanMotion → robot state trajectories
  dataset.py   episodes, actions, trimming
  export/      one module per format
  pipeline.py  ties it together
  api/         FastAPI server + job queue
  cli.py
frontend/      React + Vite UI
```

## Known limits

- **Single camera.** Depth comes from a monocular model, so absolute distances are approximate.
  Trajectories are smooth and consistent, but expect to scale or offset them for a real robot's workspace (`--scale`).
- **Not robot-specific.** States are end-effector poses and generic joint angles. There is no
  inverse kinematics, joint-limit clipping or collision checking for a particular robot yet.
- **One person, one episode per clip.** Long videos are not split into separate demonstrations.
- **No object or scene understanding.** Only the person is tracked.
- **Tracking gaps** longer than 0.25 s are filled by holding the last pose; they are flagged per frame (`tracked`) in the NPZ export.
- **LeRobot export** follows the v2.1 layout and is checked structurally, but has not been loaded with the `lerobot` library itself.
- The server has no authentication and runs one job at a time; it is meant for local or trusted use.
