"""Human motion extractors (video -> HumanMotion)."""

from .base import Extractor


def get_extractor(name: str = "mediapipe", **kwargs) -> Extractor:
    if name == "mediapipe":
        from .mediapipe_extractor import MediaPipeExtractor
        return MediaPipeExtractor(**kwargs)
    if name == "synthetic":
        from .synthetic import SyntheticExtractor
        return SyntheticExtractor(**kwargs)
    raise KeyError(f"Unknown extractor '{name}'")


__all__ = ["Extractor", "get_extractor"]
