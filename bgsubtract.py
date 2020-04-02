import pathlib
import time
from queue import Queue
from threading import Thread
from timeit import default_timer

import cv2
import numpy as np
from imutils.video import FileVideoStream
from numba import cuda

import video_stream

"""
TESTS: 

Normal loading, fast loading.
cv2 bgsubtract, averaging bgsubtract, cuda bgsubtract, numba bgsubtract

"""


class video_queue:
    """
    Video input queue

    based on this implementation:
    https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
    by Adrian Rosebrock released under MIT license
    """

    def __init__(self, file: str, queue_size: int = 500):
        """


        """
        self.stream = cv2.VideoCapture(file)
        self.stopped = False
        self.Queue = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        """

        """
        self.thread.start()
        return self

    def update(self):
        """

        """
        while True:
            if self.stopped:
                break

            if not self.Queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stopped = True
                    break
                frame = cv2.cuda_GpuMat(frame)
                self.Queue.put(frame)
            else:
                time.sleep(0.1)
        self.stream.release()

    def read(self):
        """

        """
        return self.Queue.get()

    def running(self):
        """

        """
        return self.more() or not self.stopped

    def more(self):
        """

        """
        tries = 0
        while self.Queue.qsize() == 0 and not self.stopped and tries < 20:
            time.sleep(0.1)
            tries += 1

        return self.Queue.qsize() > 0

    def stop(self):
        """

        """
        self.stopped = True
        self.thread.join()


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


def read_video(path_video: str) -> np.ndarray:
    assert pathlib.Path(path_video).exists()

    buf = []
    video_stream = FileVideoStream(path_video).start()
    while video_stream.read() is not None:
        frame = video_stream.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf.append(frame)
    return buf


def read_video_cuda(path_video: str):
    buf = []
    stream = video_stream.video_queue(path_video).start()
    while stream.more():
        frame = stream.read()

        buf.append(frame)
    return buf


def fast_bgsub_thread(filepath):
    video_stream = FileVideoStream(filepath).start()
    stacked = []
    while video_stream.read() is not None:
        frame = video_stream.read()
        # stacked[count] = frame
        # frame = imutils.resize(frame, width=450)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = np.dstack([frame, frame, frame])
        stacked.append(frame)

    return stacked


#
def normal_bgsub(filepath) -> np.ndarray:
    with Timer("normal_bgsub"):
        frames = read_video(filepath)
        # print(frames[0])
        # final_frames = []
        mask: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )
        return np.stack([mask.apply(frame) for frame in frames])


def cuda_bgsub(filepath) -> np.ndarray:
    # frames = read_video_cuda(filepath)
    stream = cv2.cuda_Stream()

    # cv2.createB
    mask: cv2.cuda.createBackgroundSubtractorMOG2 = cv2.cuda.createBackgroundSubtractorMOG2(
        history=3, varThreshold=100, detectShadows=False
    )
    get_frames = video_queue(filepath).start()
    post_process = []
    with Timer("cuda_bgsub"):
        while get_frames.more():
            frame = get_frames.read()
            processed = mask.apply(frame, -1, stream)
            post_process.append(processed)
    return processed

@cuda.jit
def numba_bgsub(filepath):
    #implement custom background subtract with numba
    pass

def main():
    filepath = "Raw Video Output 10x Inv-L.avi"

    normal_bgsub(filepath)

    cuda_bgsub(filepath)


if __name__ == "__main__":
    main()
