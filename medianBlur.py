import pathlib
from timeit import default_timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from imutils.video import FileVideoStream
from pyvips import Image

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


class helper:
    def read_video(self,stream) -> np.ndarray:
        buf = []
        while stream.more():
            frame = stream.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf.append(frame)
        return buf

    def normal_bgsub(self, stream) -> np.ndarray:

        frames = self.read_video(stream)
        mask: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )
        return np.stack([mask.apply(frame) for frame in frames])

    def write_frames(self, frames: list, path, cuda=False) -> None:
        root = pathlib.Path(path)
        if not root.exists():
            root.mkdir()
        if cuda:
            root = pathlib.Path("output_cuda")
            if not root.exists():
                root.mkdir()
            for i, f in enumerate(frames):
                plt.imsave(str(root / f"{i}.jpg"), f.download())
            return 0
        else:
            for i, f in enumerate(frames):
                plt.imsave(str(root / f"{i}.jpg"), f)

    def get_functions(self):
        return cv2.cuda.createMedianFilter(cv2.CV_8UC1, 3), cv2.cuda.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )



def norm_medianFilter(frames):
    return np.stack([cv2.medianBlur(frame, 3) for frame in frames])

def cuda_median(video_stream, filter: cv2.cuda_Filter, mask: cv2.cuda.createBackgroundSubtractorMOG2, gaussian=False) -> list:
    mask = cv2.cuda.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )
    post_process = []
    total = 0
    cuda_stream = cv2.cuda_Stream()
    if gaussian:
        filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (3,3), 0.5)
    while video_stream.more():
    # processed = mask.apply(frame, -1, stream)
        frame = video_stream.read()
        if frame is None:
            break
        print(frame, mask, cuda_stream)
        processed = mask.apply(frame, 0.009, cuda_stream)
        filter.apply(processed)
        post_process.append(processed)
    return post_process

def pillow_filter(frames) -> list:
    # video_stream = FileVideoStream(filepath).start()
    median_filter = ImageFilter.MedianFilter(3)
    image_arr = []

    for frame in frames:
        image = Image.fromarray(frame).filter(median_filter)
        image_arr.append(image)
    return image_arr

def pyvips_filter(video_stream) -> list:
    mask = cv2.createBackgroundSubtractorMOG2(history=3, varThreshold=100, detectShadows=False)
    ret_buf = []
    while video_stream.more():
        frame = video_stream.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = mask.apply(frame)
        height, width = frame.shape
        linear = frame.reshape(width * height)
        vips_image = pyvips.Image.new_from_memory(linear.data, width, height, 1, "uchar")
        ret_buf.append(vips_image.median(3))
#        ret_buf.append(frame)
    video_stream.stop()
    return ret_buf

def main():
    #init memory
    path_to_video = "Raw Video Output 10x Inv-L.avi"
    video_stream = FileVideoStream(path_to_video).start()
    cuda_stream = cuda_video_stream.video_queue(path_to_video, 0, 0, 934, 239).start()
    # #benchmark

    helper_class = helper()
    filter, mask = helper_class.get_functions()

    with Timer("cuda"):
        frames = cuda_median(cuda_stream, filter, mask)
    cuda_stream.stop()

    with Timer("cv2"):
        frames = helper_class.normal_bgsub(video_stream)
        filtered = norm_medianFilter(frames)
    video_stream.stop()

    video_stream = FileVideoStream(path_to_video).start()
    with Timer("PIL test"):
        frames = helper_class.normal_bgsub(video_stream)
        output = pillow_filter(frames)
    video_stream.stop()

    video_stream = FileVideoStream(path_to_video).start()
    with Timer("Pyvips test"):
        output = pyvips_filter(video_stream)
    video_stream.stop()

if __name__ == '__main__':
    main()
