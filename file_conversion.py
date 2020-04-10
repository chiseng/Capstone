import pathlib
import time

import cv2
import numpy as np
from imutils.video import FileVideoStream


class Timer:
    def __init__(self, label, decimals=1, announce_start=False):
        self.announce_start = announce_start
        self.label = label
        self.decimals = decimals

    def get_printout(self, message: str) -> str:
        return str({self.__class__.__name__: (self.label, message)})

    def __enter__(self):
        self.start = time.time()
        if self.announce_start:
            print(self.get_printout("start"))

    def __exit__(self, type, value, traceback):
        duration = time.time() - self.start
        duration = round(duration, self.decimals)
        print(self.get_printout(f"{duration} s"))


def read_video(path_video: str) -> np.ndarray:
    assert pathlib.Path(path_video).exists()

    buf = []
    video_stream = FileVideoStream(path_video).start()
    while video_stream.read() is not None:
        frame = video_stream.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf.append(frame)
    return buf


def read_video(path_video: str) -> np.ndarray:
    assert pathlib.Path(path_video).exists()
    cap = cv2.VideoCapture(path_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_count, height, weight, 3), np.uint8)

    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf[fc] = frame
        fc += 1
    cap.release()
    return buf

def cv2_background_subtract(frames: np.ndarray) -> np.ndarray:
    model = cv2.createBackgroundSubtractorMOG2(history=3, varThreshold=100, detectShadows=False)
    return np.stack([model.apply(f) for f in frames])

def write_video_avi(frames: np.ndarray, fps: int):
    out_file = "test_vid.avi"
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('X','V','I','D'), fps, frameSize=(frames.shape[2], frames.shape[1]), isColor=False)
    for i, f in enumerate(frames):
        out.write((f))
    out.release()



def main(path_video_in="Raw Video Output 10x Inv-L.avi", out_dir="output_frames"):
    v = cv2.VideoCapture(path_video_in)
    v.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    fps = int(v.get(cv2.CAP_PROP_FPS))
    video = read_video(path_video_in)
    output = cv2_background_subtract(video)
    print("Frame conversion after background subtraction for demonstration")
    with Timer("avi conversion",decimals=7):
        write_video_avi(output, fps)
    with Timer("NPY file save", decimals=7):
        np.save("video_frames", output)


if __name__ == "__main__":
    main()
