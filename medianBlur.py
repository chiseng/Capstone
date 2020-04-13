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
    while stream.more() and stream.read() is not None:
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


def write_frames(frames: list, cuda=False) -> None:
    root = pathlib.Path("output_cv2")
    if not root.exists():
        root.mkdir()
    if cuda:
        root = pathlib.Path("output_cuda")
        if not root.exists():
            root.mkdir()
        for i, f in enumerate(frames):
            plt.imsave(str(root / f"{i}.jpg"),f.download())
        return 0#, f.download())
    else:
        for i, f in enumerate(frames):
            plt.imsave(str(root / f"{i}.jpg"),f)

def standard_medianfilter(frames: list):
    # frames = read_video(filepath)
    return (np.stack([cv2.blur(frame,(3,3)) for frame in frames]) > 0.5).astype(np.uint8)

def cuda_median(frames, stream, filter: cv2.cuda_Filter, mask):
    post_process = []
    total = 0
    while frames.more() and frames.read() is not None:
    # processed = mask.apply(frame, -1, stream)
        frame:cv2.cuda_GpuMat = frames.read()
        processed = mask.apply(frame, 0.009, stream)
        filter.apply(processed)
        post_process.append(processed)
    return post_process

def get_functions():
    return cv2.cuda.createBoxFilter(cv2.CV_8UC1,cv2.CV_8UC1,(3,3),borderMode=cv2.BORDER_CONSTANT), cv2.cuda.createBackgroundSubtractorMOG2(
        history=3, varThreshold=100, detectShadows=False
    ), cv2.cuda_Stream()

def main():
    path_to_video = "Raw Video Output 10x Inv-L.avi"
    #init memory

    video_stream = FileVideoStream(path_to_video).start()
    filter, mask, stream = get_functions()
    #benchmark

    with Timer("cv2"):
        frames = normal_bgsub(video_stream)
        filtered = norm_medianFilter(frames)
    write_frames(filtered)
    get_frames = cuda_video_stream.video_queue(path_to_video, 0, 0, 934, 239).start()
    with Timer("cv2 with cuda median filter"):
        ret_arr = cuda_median(get_frames, stream, filter, mask)
    write_frames(ret_arr, cuda=True)

    # with Timer("cv2 with cuda"):
    #     ret_arr = cuda_median(get_frames, stream)
    #     print(ret_arr)

if __name__ == '__main__':
    main()
