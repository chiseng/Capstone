import pathlib
from timeit import default_timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import FileVideoStream
from numba.decorators import jit

import cuda_video_stream

"""
TESTS: 

Normal loading, fast loading.
cv2 bgsubtract, cv2 with cuda bgsubtract, numba bgsubtract

"""


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
    # assert pathlib.Path(path_video).exists()
    count = 0
    buf = []
    while stream.more() and stream.read() is not None:
        frame = stream.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf.append(frame)
    stream.stop()
    return buf


#
def normal_bgsub(video_stream) -> np.ndarray:

    frames = read_video(video_stream)
    mask: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
        history=3, varThreshold=100, detectShadows=False
    )
    return np.stack([mask.apply(frame) for frame in frames])


def cuda_bgsub(get_frames) -> np.ndarray:
    stream = cv2.cuda_Stream()

    # cv2.createB
    mask: cv2.cuda.createBackgroundSubtractorMOG2 = cv2.cuda.createBackgroundSubtractorMOG2(
        history=3, varThreshold=100, detectShadows=False
    )

    post_process = []

    while get_frames.more() and get_frames.read() is not None:
        frame = get_frames.read()
        processed = mask.apply(frame,-1, stream)
        post_process.append(processed)
    return post_process


@jit(nopython=True)
def numba_bgsub(frames, threshold):

    """
    Since numba works better with more verbose implemenation of methods,
    we expand out the calculation of mean values with mutliple for loops.
    """
    # find sum of all the pixels [c,w,h]

    len_frames = frames.shape[0]
    h = frames.shape[1]
    w = frames.shape[2]
    c = frames.shape[3]
    avg_frame = np.zeros((h, w, c))
    for i in range(len_frames):
        for j in range(h):
            for k in range(w):
                for l in range(c):
                    avg_frame[j, k, l] += frames[i][j, k, l]

    for j in range(h):
        for k in range(w):
            for l in range(c):
                avg_frame[j, k, l] /= len_frames

    for i in range(len_frames):
        for j in range(h):
            for k in range(w):
                for l in range(c):
                    frames[i][j, k, l] = np.abs(frames[i][j, k, l] - avg_frame[j, k, l])

    diff_mean = np.zeros((len_frames, h, w))
    for i in range(len_frames):
        for j in range(h):
            for k in range(w):
                interm = 0
                for l in range(c):
                    interm += frames[i][j, k, l]
                diff_mean[i, j, k] = interm / float(c)

    mask = (diff_mean > threshold).astype(np.uint8)

    return mask


def numba_test(video: str):
    frames = read_video(video)
    output = numba_bgsub(np.stack(frames), 100)
    # print(output)
    return output


def write_frames(frames: np.ndarray, cv2=False) -> None:
    root = pathlib.Path("output_norm")
    if not root.exists():
        root.mkdir()
    for i, f in enumerate(frames):
        plt.imsave(str(root / f"{i}.jpg"), f)


def main():
    filepath = "Raw Video Output 10x Inv-L.avi"
    #Load frames into memory
    get_frames = cuda_video_stream.video_queue(filepath, 0, 0, 934, 239).start()
    stream = FileVideoStream(filepath, queue_size=500).start()

    #Benchmark tests
    with Timer("OpenCV"):
        output1 = normal_bgsub(stream)
    with Timer("OpenCV with CUDA"):
        output2 = cuda_bgsub(get_frames)
    print(len(output2))
    print(len(output1))
    stream = FileVideoStream(filepath).start()
    with Timer("Numba test"):
       output = numba_test(stream)





if __name__ == "__main__":
    main()
