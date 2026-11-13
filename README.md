# footage2robot

Turn videos of people doing things into training data for robots.

Record a person performing a task on your phone (or upload footage). Each demonstration
becomes an episode: the body and hands are tracked in 3D, converted to robot states and
actions — generic or for a specific robot — and exported in the dataset formats
robot-learning code already reads.

```
phone / upload ─► tracking ─► clean-up ─► calibration ─► episodes ─► retargeting ─► exporters
                  (body + 2    (gaps,      (height,       (split,     (robot IK,     (7 formats)
                   hands, 3D)   smoothing)  marker)        trim)       joint limits)
```

## What it produces

**Generic robot types** — robot-independent states:

| Key | State | Notes |
| --- | --- | --- |
| `single_arm` | 7-D: end-effector xyz, rotation, gripper | most active hand, or pick left/right |
| `bimanual` | 14-D: both arms | |
| `dex_hand` | 25-D: wrist pose + 19 finger joint angles | 15 flexion + 4 spread |
| `humanoid` | 25-D: waist, head, arms, legs, hand closure | generic joint tree, radians |
| `keypoints` | 227-D: raw 3D body + hand keypoints | for representation learning / custom retargeting |

Poses are expressed in a body-centric frame (x forward, y left, z up, origin between the
shoulders), so the result does not depend on where the camera was — or, with a workspace
marker, in the marker's frame. Gripper is `0` open → `1` closed.
Actions are either the next state (`absolute`) or the change to it (`delta`).

**Specific robots** — real joint commands, solved on the robot's model and kept inside its joint limits:

| Key | Robot | How |
| --- | --- | --- |
| `so101` | SO-101 / SO-100 (LeRobot arm), 5 joints + gripper | inverse kinematics |
| `franka_panda` | Franka Panda / FR3, 7 joints + gripper | inverse kinematics |
| `ur5e` | Universal Robots UR5e, 6 joints | inverse kinematics |
| `unitree_g1` | Unitree G1 humanoid, 29 joints + hands | limb-angle mapping, clipped to limits |
| `custom` | any serial arm | `--urdf robot.urdf --base-link ... --ee-link ...` |

Arms follow the hand: its motion relative to the person's shoulder is scaled to the robot's reach
and replayed from the robot's shoulder, lifted just enough to stay above the robot's table.
Every dataset reports the IK position error and how many frames were reachable.

**Formats**:

| Key | Format | Used by |
| --- | --- | --- |
| `lerobot` | LeRobot v3.0 (Parquet + MP4) | current Hugging Face LeRobot: ACT, Diffusion Policy, pi0, SmolVLA |
| `lerobot_v21` | LeRobot v2.1 | older LeRobot releases and tools |
| `rlds` | RLDS / TFDS (TFRecord) | Open X-Embodiment, Octo, OpenVLA |
| `robomimic` | robomimic HDF5 | robomimic, RoboSuite, MimicGen |
| `zarr` | Diffusion Policy replay buffer | Diffusion Policy, UMI |
| `npz` | NumPy per episode + JSON | anything |
| `json` | Plain JSON, no images | inspection, web tools |

## Features

- **Record on a phone.** The web app opens the rear camera in the browser; press record once
  per demonstration. Browsers only allow the camera on https pages (see below).
- **Real-world scale.** Enter the person's height or shoulder-to-wrist length and the skeleton
  is scaled to true metres. Print the workspace marker (`footage2robot marker`), lay it on the
  table, and positions are expressed relative to it.
- **Episodes.** One take is one episode; long clips can be split automatically where the hands
  pause with nothing held, and every cut can be adjusted in the editor without re-tracking.
- **Objects.** Click the handled object once and it is followed through the clip; its image
  position is exported as `observation.environment_state`.
- **Accounts.** Optional sign-in with Firebase Authentication; each user sees only their own jobs.
- **Queue.** Jobs live in a SQLite table and are run by separate worker processes.

## Run it

### Docker

```bash
docker compose up --build        # http://localhost:8000
```

The image pins Python 3.11 (needed by the RLDS exporter) and bundles ffmpeg, the graphics
libraries and the tracking models.

### From source

Requirements: Python 3.10–3.12 (3.13 works for everything except RLDS), Node 18+, `ffmpeg`,
and on headless Linux `apt-get install libegl1 libgles2`.

```bash
cd backend && pip install -e ".[dev,rlds]"
cd ../frontend && npm install && npm run build
cd ../backend && footage2robot serve --workers 2      # http://127.0.0.1:8000
```

For frontend development run `npm run dev` (port 5173, proxies `/api` to port 8000).

### Using a phone camera

The phone must load the app over https:

- **Same Wi-Fi:** `footage2robot serve --host 0.0.0.0 --https` creates a self-signed
  certificate. Open `https://<computer-ip>:8000` on the phone and accept the warning.
- **Anywhere:** put the server behind a tunnel or reverse proxy that provides https
  (for example `cloudflared tunnel --url http://localhost:8000`), or deploy the Docker image
  to a host that terminates TLS.

### Accounts (Firebase)

1. In the Firebase console create a project, add a **Web app**, and under
   Authentication → Sign-in method enable **Email/Password** and/or **Google**.
2. Under Authentication → Settings → Authorized domains, add the domain you serve the app from.
3. Start the server with the two settings from `.env.example`:

```bash
export FOOTAGE2ROBOT_FIREBASE_PROJECT=your-project-id
export FOOTAGE2ROBOT_FIREBASE_WEB_CONFIG='{"apiKey":"...","authDomain":"...","projectId":"...","appId":"..."}'
```

With these set, every API call needs a signed-in user and jobs are private to their owner.
Without them the server runs as a single local user — only do that on a machine you trust.

### Command line

```bash
footage2robot convert demo1.mp4 demo2.mp4 -o out \
  --task "pick up the cup and place it on the shelf" \
  --embodiments single_arm,so101 --formats lerobot,rlds \
  --height 1.75 --split auto --track-objects

footage2robot convert demo.mp4 -o out --marker-size 0.10 --base-offset 0.3 0 0   # marker frame
footage2robot convert demo.mp4 -o out --urdf my_arm.urdf --base-link base --ee-link tool0
footage2robot marker             # write the printable workspace marker
footage2robot formats            # list embodiments and formats
```

Output:

```
out/summary.json, tracking.json
out/motion/episode_000.npz                  tracked human motion
out/datasets/<embodiment>/<format>/...      exported datasets
```

### API

| Method | Path | |
| --- | --- | --- |
| `GET` | `/api/config` | whether sign-in is required + Firebase web config |
| `GET` | `/api/options` | embodiments, robots, formats, limits |
| `POST` | `/api/jobs` | multipart `files[]` + `config` (JSON) → job |
| `POST` | `/api/jobs/demo` | run on generated motion, no footage needed |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | status, progress, summary |
| `POST` | `/api/jobs/{id}/edits` | change episode cuts / object point, then re-export |
| `GET` | `/api/jobs/{id}/clips/{n}/overlay` | 2D keypoints, episode cuts, object box |
| `GET` | `/api/jobs/{id}/datasets/{embodiment}/preview` | state trajectories |
| `GET` | `/api/jobs/{id}/download?embodiment=&format=` | zip of one export, or everything |
| `DELETE` | `/api/jobs/{id}` | |
| `GET` | `/api/marker.png` | printable workspace marker |

Requests carry `Authorization: Bearer <Firebase ID token>` when accounts are on. Video and
download links can instead carry the job's `access_key` (`?key=`), so they work in `<video>` and `<a>`.

Settings: `FOOTAGE2ROBOT_DATA_DIR` (job storage, default `./f2r_data`),
`FOOTAGE2ROBOT_MAX_UPLOAD_MB` (default 2048), `FOOTAGE2ROBOT_MAX_ACTIVE_JOBS` (per user, default 3),
`FOOTAGE2ROBOT_MODEL_DIR` (model cache).

## Tests

```bash
cd backend && pytest
```

The suite runs the whole pipeline on generated motion and small rendered clips, loads every
export back with its own reader, checks robot kinematics against published poses, and exercises
accounts, the queue and edits through the API.

To also load the LeRobot export with the real `lerobot` library, point the tests at an
interpreter that has it (it pins dependencies that clash with this project's, so use a
separate virtualenv):

```bash
python -m venv /tmp/lr && /tmp/lr/bin/pip install "lerobot[dataset]"
FOOTAGE2ROBOT_LEROBOT_PYTHON=/tmp/lr/bin/python pytest tests/test_export.py
```

## Layout

```
backend/footage2robot/
  extract/      video → HumanMotion (MediaPipe pose + hands; synthetic generator for tests)
  process/      gap filling, smoothing
  calibrate.py  body-measurement scale, camera model, workspace marker
  segment.py    episode splitting
  objects.py    object tracking (CPU tracker, optional SAM 2)
  retarget/     HumanMotion → generic robot states
  robots/       URDF kinematics, inverse kinematics, robot profiles, bundled models
  dataset.py    episodes, actions, trimming
  export/       one module per format
  pipeline.py   track stage + build stage
  api/          FastAPI server, Firebase token checks, SQLite queue, worker
  cli.py
frontend/       React + Vite UI (recorder, episode editor, sign-in)
Dockerfile, docker-compose.yml
```

## Known limits

- **Depth is still estimated from one camera.** A body measurement fixes overall scale; the marker
  fixes the workspace origin but assumes a typical phone field of view (`fov_deg`, default 65°) and a
  camera that does not move. True per-pixel depth needs a depth sensor (e.g. a native iPhone LiDAR app,
  not part of this project yet).
- **Robot arms are mapped by a heuristic**, not a calibrated scene: without a marker, the hand's motion
  is scaled to the robot's reach. No collision checking. Check trajectories in simulation before running
  them on hardware.
- **Unitree G1** output is joint angles only: the pelvis pose and balance are not estimated, so it is not
  directly executable as whole-body motion.
- **SO-101 joint angles are in radians** on the URDF's zero pose; LeRobot's own SO-101 recordings use
  calibrated motor units, so a conversion is needed before mixing the two.
- **Object tracking is 2D** (a box in the image). The built-in CPU tracker can drift when the hand covers
  the object. The SAM 2 backend (`--object-backend sam2`, `pip install -e ".[sam2]"`) is written against
  SAM 2's documented API but has not been run here (its weights could not be downloaded).
- **Episode auto-splitting** relies on pauses; back-to-back demonstrations without a pause need manual cuts.
- **Tracking gaps** longer than 0.25 s are filled by holding the last pose; they are flagged per frame
  (`tracked`) in the NPZ export.
- **Docker image** has not been built in this development environment (no Docker daemon was available).
- **Storage** is the local disk and SQLite, which suits one server. Several servers would need shared
  storage and a hosted database.
