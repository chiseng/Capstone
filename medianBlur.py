import pathlib
from timeit import default_timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import FileVideoStream

import cuda_video_stream


class Timer(object):
    def __init__(self, method):
        self.method = method
        self.timer = default_timer

    def __enter__(self):
        print(f"Starting timer for {self.method} function")
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs
        print("elapsed time: %f s" % self.elapsed)

def read_video(stream) -> np.ndarray:

    buf = []
    while stream.read() is not None:
        frame = stream.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf.append(frame)
    return buf


def normal_bgsub(stream) -> np.ndarray:

    frames = read_video(stream)
    mask: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
        history=3, varThreshold=100, detectShadows=False
    )
    return np.stack([mask.apply(frame) for frame in frames])

def norm_medianFilter(frames):
    return np.stack([cv2.medianBlur(frame, 3) for frame in frames])


def write_frames(frames: np.ndarray, cv2=False) -> None:
    root = pathlib.Path("output_numba")
    if not root.exists():
        root.mkdir()
    for i, f in enumerate(frames):
        plt.imsave(str(root / f"{i}.jpg"),f) #, f.download())

def standard_medianfilter(frames: list):
    # frames = read_video(filepath)
    return (np.stack([cv2.blur(frame,(3,3)) for frame in frames]) > 0.5).astype(np.uint8)

def cuda_median(frames, stream, filter):
    # cv2.createB
    mask: cv2.cuda.createBackgroundSubtractorMOG2 = cv2.cuda.createBackgroundSubtractorMOG2(
        history=3, varThreshold=100, detectShadows=False
    )

    post_process = []
    # for frame in frames:
    #     processed = mask.apply(frame, -1, stream)
    #     post_process.append(processed)
    # frame = frames.read()
    while frames.more():
        # processed = mask.apply(frame, -1, stream)
        frame = frames.read()
        filter.apply(frame)
        processed = mask.apply(frame, -1, stream)
        post_process.append(processed)
    return post_process


def main():
    path_to_video = "Raw Video Output 10x Inv-L.avi"
    #init memory
    stream = cv2.cuda_Stream()
    get_frames = cuda_video_stream.video_queue(path_to_video).start()
    get_frames.stop()
    get_frames = cuda_video_stream.video_queue(path_to_video).start()
    video_stream = FileVideoStream(path_to_video).start()
    video_stream.stop()
    video_stream = FileVideoStream(path_to_video).start()

    #benchmark
    with Timer("cv2 with cuda"):
        ret_arr = cuda_median(get_frames, stream)

    with Timer("cv2"):
        frames = normal_bgsub(video_stream)
        filtered = norm_medianFilter(frames)

    with Timer("cv2 with cuda"):
        ret_arr = cuda_median(get_frames, stream)

if __name__ == '__main__':
    main()
