"""RLDS export (TFDS on-disk format, as used by Open X-Embodiment).

Each episode is a record with a nested ``steps`` dataset following the RLDS
spec (observation, action, reward, discount, is_first, is_last, is_terminal,
language_instruction). Load with::

    import tensorflow_datasets as tfds
    ds = tfds.builder_from_directory("<export dir>/<name>/1.0.0").as_dataset(split="train")

Requires the optional ``rlds`` extra (tensorflow + tensorflow-datasets).
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np

from ..dataset import RobotDataset


class RLDSUnavailable(RuntimeError):
    pass


def _import_tfds():
    try:
        import tensorflow  # noqa: F401
    except ImportError as e:
        raise RLDSUnavailable("RLDS export needs TensorFlow: pip install 'footage2robot[rlds]'") from e
    # tensorflow-datasets hard-imports the unmaintained `promise` package (only used by
    # its downloader), which no longer builds on recent Pythons. Shim it if absent.
    try:
        import promise  # noqa: F401
    except ImportError:
        shim = types.ModuleType("promise")
        shim.Promise = type("Promise", (), {})
        sys.modules["promise"] = shim
    try:
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise RLDSUnavailable("RLDS export needs tensorflow-datasets: pip install 'footage2robot[rlds]'") from e
    return tfds


def export_rlds(ds: RobotDataset, out_dir: Path) -> Path:
    tfds = _import_tfds()
    from tensorflow_datasets.core import dataset_builders

    d = ds.state_dim
    obs = {"state": tfds.features.Tensor(shape=(d,), dtype=np.float32, doc=", ".join(ds.state_names))}
    if ds.has_env_state:
        obs["object"] = tfds.features.Tensor(shape=(len(ds.env_state_names),), dtype=np.float32,
                                             doc=", ".join(ds.env_state_names))
    w, h = ds.image_size
    if ds.has_images:
        obs["image"] = tfds.features.Image(shape=(h, w, 3), dtype=np.uint8, encoding_format="jpeg")
    features = tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "observation": tfds.features.FeaturesDict(obs),
            "action": tfds.features.Tensor(shape=(d,), dtype=np.float32, doc=", ".join(ds.action_names)),
            "reward": tfds.features.Scalar(dtype=np.float32),
            "discount": tfds.features.Scalar(dtype=np.float32),
            "is_first": tfds.features.Scalar(dtype=np.bool_),
            "is_last": tfds.features.Scalar(dtype=np.bool_),
            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
            "language_instruction": tfds.features.Text(),
        }),
        "episode_metadata": tfds.features.FeaturesDict({
            "file_path": tfds.features.Text(),
            "embodiment": tfds.features.Text(),
            "tracked_fraction": tfds.features.Scalar(dtype=np.float32),
        }),
    })

    def episodes():
        for i, ep in enumerate(ds.episodes):
            n = len(ep)
            images = ep.iter_images(ds.image_size) if ds.has_images else None
            steps = []
            for t in range(n):
                o = {"state": ep.state[t]}
                if ds.has_env_state:
                    o["object"] = ep.env_state[t]
                if images is not None:
                    o["image"] = next(images)
                steps.append({
                    "observation": o, "action": ep.action[t],
                    "reward": np.float32(t == n - 1), "discount": np.float32(1.0),
                    "is_first": t == 0, "is_last": t == n - 1, "is_terminal": t == n - 1,
                    "language_instruction": ep.task,
                })
            yield f"episode_{i:06d}", {"steps": steps, "episode_metadata": {
                "file_path": ep.video_path or "", "embodiment": ds.embodiment,
                "tracked_fraction": np.float32(ep.meta.get("tracked_fraction", 1.0))}}

    name = re.sub(r"[^a-z0-9_]+", "_", ds.name.lower()).strip("_") or "footage2robot"
    if not name[0].isalpha():
        name = f"ds_{name}"
    dataset_builders.store_as_tfds_dataset(
        name=name, version="1.0.0", features=features, split_datasets={"train": episodes()},
        data_dir=str(out_dir),
        description=f"Human video demonstrations retargeted to '{ds.embodiment}' by footage2robot. "
                    f"fps={ds.fps}, action_mode={ds.action_mode}.",
    )
    return out_dir / name / "1.0.0"
