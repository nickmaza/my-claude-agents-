import cv2
import numpy as np
import pytest

from footage2robot.extract import get_extractor
from footage2robot.process import clean_motion


@pytest.fixture(scope="session")
def motion():
    return clean_motion(get_extractor("synthetic").extract(""))


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(45):
        frame = np.full((120, 160, 3), 30, np.uint8)
        cv2.circle(frame, (20 + 2 * i, 60), 10, (0, 200, 255), -1)
        w.write(frame)
    w.release()
    return str(path)
