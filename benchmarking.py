import math
import time

import cv2
import numpy as np
from imutils.video import FPS
from imutils.video import FileVideoStream
# import tkinter as tk
# from tkinter.filedialog import askopenfilenames
# from tkinter.filedialog import asksaveasfilename
from pandas import DataFrame

import cuda_video_stream as cvs

### ----- Parameters to Change ----- ###
H = 140  # No. of pixels to select for height of Region Of Interest
blur_value = (
    7  # value = 3,5 or 7 (ODD).median Blur value determines the accuracy of detection
)
Delay = 1000  # time value in miliseconds. (Fastest) Minimum = 1ms
Show = 1  # To display the image. 1 = On, 0 = Off
Skip_frames = 20  # number of frames to skip before Im showing
file_name = "Raw Video Output 10x Inv-L.avi"  # Getting all open files location
Channels = 25
line_color = (200, 100, 100)


### ----- Parameters to Change ----- ###


"""
We test with the one avi file to benchmark
"""


# def crop(frame,roi):

format_to_dtype = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}


def to_crop(frame, r, Channels):
    ch = Channels
    # x,y represents the coordinates of the upper most corner of the rectangle
    print("[ROI] (x , y, width, height) is", r)

    # Crop image
    y1 = int(r[1])  # y
    y2 = int(r[1] + r[3])  # y + height = height of cropped
    x1 = int(r[0])  # x
    x2 = int(r[0] + r[2])  # x + width = width of cropped

    print(x1, x2, y1, y2)
    imCrop = frame[y1 : (y1 + H), x1:x2]
    print(frame.shape)
    # the array for sub-channels
    sub_ch = []

    # draw lines on crop frame
    for x in range(ch + 1):
        sub_ch_x = round(
            x * (r[2] / (ch))
        )  # place where line will be drawn, proportional to width
        sub_ch.append(sub_ch_x)
    #     cv2.line(imCrop, (sub_ch[x], 0), (sub_ch[x], H), line_color, 1)
    return x1, x2, y1, y2, sub_ch


def save_excel(sum_ch1):
    total_sum = []
    total_sum.append(sum_ch1)

    ###write dataframes and export to an Excel file
    check = 0
    title = []
    for j in range(len(total_sum)):
        if check < len(total_sum[j]):
            check = len(total_sum[j])
        title.append("Run 1")

    index = np.arange(0, check, 1)

    for k in range(len(total_sum)):
        if len(total_sum[k]) < check:
            for l in range(len(total_sum[k]), check):
                total_sum[k].append(0)

    TTotal_sum = list(map(list, zip(*total_sum)))
    # print(TTotal_sum)
    df = DataFrame(data=TTotal_sum, columns=title)
    # savefile = asksaveasfilename(filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
    df.to_excel("testfile" + ".xlsx", index=False, sheet_name="Results")


class CUDA:
    def __init__(self, file_name, y1, y2, x1, x2, sub_ch):
        self.mask: cv2.cuda.createBackgroundSubtractorMOG2 = cv2.cuda.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )
        self.filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (blur_value, blur_value), 0.5
        )
        self.stream = cv2.cuda_Stream()
        self.gpu_read = cvs.video_queue(file_name, y1, y2, x1, x2).start()
        self.sum_ch1 = np.zeros(28)
        self.blur_bgsub = {}
        self.contour_detection = {}
        self.median_blur = []
        self.bgsub = []
        self.sub_ch = sub_ch
        self.thresh = []

    def cuda_run(self):

        count = 0
        fps = FPS().start()
        start = time.time()
        while self.gpu_read.more():
            cycle_start = time.time()
            augment_start = time.time()
            frame = self.gpu_read.read()
            bg = time.time()
            aug_frame = self.mask.apply(frame, 0.009, self.stream)
            bg_stop = time.time()
            self.bgsub.append(bg_stop - bg)

            blur = time.time()
            self.filter.apply(aug_frame)
            blur_stop = time.time()
            self.median_blur.append(blur_stop - blur)

            crop = aug_frame.download()
            threshh = time.time()
            crop = cv2.threshold(crop, 125, 255, cv2.THRESH_BINARY)[1]
            thresh_stop = time.time()
            self.thresh.append(thresh_stop - threshh)
            augment_end = time.time()
            self.blur_bgsub[count] = augment_end - augment_start
            count += 1

            count_start = time.time()
            contours, hierarchy = cv2.findContours(
                crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # list of all the coordinates (tuples) of each cell
            # print(contours)
            # to find the coordinates of the cells
            for i in range(len(contours)):
                avg = np.mean(contours[i], axis=0)
                coord = (int(avg[0][0]), int(avg[0][1]))  ##Coord is (y,x)
                ch_pos = int(math.floor((coord[0]) / self.sub_ch[1]))
                try:
                    self.sum_ch1[ch_pos] += float(1)
                except:
                    pass
            count_end = time.time()
            self.contour_detection[count] = count_end - count_start

            fps.update()
            cycle_end = time.time()
        print(count)
        self.gpu_read.stop()
        fps.stop()
        print("[INFO] Each cycle time taken = %0.5fs" % (cycle_end - cycle_start))
        print(f"[RESULTS] for RUN is {self.sum_ch1}")
        return (
            self.blur_bgsub,
            self.median_blur,
            self.bgsub,
            self.thresh,
            self.contour_detection,
            fps,
        )


class standard:
    def __init__(self, filename):
        self.mask: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )

        self.sum_ch1 = np.zeros(28)
        self.blur_bgsub = {}
        self.contour_detection = {}
        self.median_blur = []
        self.bgsub = []
        self.thresh = []
        self.count = 0

    def vips_to_array(self,vi):
        return np.ndarray(
                buffer=vi.write_to_memory(),
            dtype=format_to_dtype[vi.format],
            shape=[vi.height, vi.width, vi.bands],
        )

    def vips_filter(self,frame):
        height, width = frame.shape
        linear = frame.reshape(width * height)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, 1, "uchar")
        filtered = vi.median(3)
        image = vips_to_array(filtered)
        return image

    def standard_run(self, x1: int, x2: int, y1: int, y2: int, sub_ch: list, pyvips=False):
        fps = FPS().start()
        start = time.time()
        cap = FileVideoStream(file_name).start()
        while cap.more():
            cycle_start = time.time()
            frame = cap.read()
            if frame is None:
                break
            frame = frame[y1:y2, x1:x2]
            augment_start = time.time()
            # crop = bgSubtract(mask,pic)
            bg = time.time()
            crop = self.mask.apply(frame)
            bg_stop = time.time()
            self.bgsub.append(bg_stop - bg)

            blur = time.time()
            if pyvips:
                img = self.vips_filter(crop)
                crop = self.vips_to_array(img)
            else:
                cv2.medianBlur(crop, 3)
            blur_stop = time.time()
            self.median_blur.append(blur_stop - blur)

            threshh = time.time()
            crop = cv2.threshold(crop, 125, 255, cv2.THRESH_BINARY)[1]
            thresh_stop = time.time()
            self.thresh.append(thresh_stop - threshh)
            augment_end = time.time()
            self.blur_bgsub[self.count] = augment_end - augment_start
            self.count += 1

            count_start = time.time()
            contours, hierarchy = cv2.findContours(
                crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # list of all the coordinates (tuples) of each cell
            # print(contours)
            # to find the coordinates of the cells
            for i in range(len(contours)):
                avg = np.mean(contours[i], axis=0)
                coord = (int(avg[0][0]), int(avg[0][1]))  ##Coord is (y,x)
                ch_pos = int(math.floor((coord[0]) / sub_ch[1]))
                try:
                    self.sum_ch1[ch_pos] += float(1)
                except:
                    pass
            count_end = time.time()
            self.contour_detection[self.count] = count_end - count_start

            fps.update()
            cycle_end = time.time()
        print(self.count)
        end = time.time()
        fps.stop()
        detect_benchmark = end - start
        print("Time taken for counting:", detect_benchmark)
        print("[INFO] Each cycle time taken = %0.5fs" % (cycle_end - cycle_start))
        print(f"[RESULTS] for RUN is {self.sum_ch1}")
        return (
            self.blur_bgsub,
            self.median_blur,
            self.bgsub,
            self.thresh,
            self.contour_detection,
            fps,
        )


def main(cuda=False, pyvips=False):
    # Get ROI from frames
    cap = FileVideoStream(file_name).start()
    image = cap.read()
    print("***** PROCESSING ROI for RUN 1 ***** File: %s" % file_name)
    cap.stop()
    """
    Get crop size and draw lines
    """
    print("***** PROCESSING RUN 1 ***** File: %s" % file_name)

    r = [0, 0, 934, 239]
    x1, x2, y1, y2, sub_ch = to_crop(image, r, Channels)

    """
    Background Subtract
    """

    # run count
    if cuda:
        run_cuda = CUDA(file_name, y1, y2, x1, x2, sub_ch)
        (
            blur_bgsub,
            median_blur,
            bgsub,
            thresh,
            contour_detection,
            fps,
        ) = run_cuda.cuda_run()
    if pyvips:
        run_standard = standard(file_name)
        (
            blur_bgsub,
            median_blur,
            bgsub,
            thresh,
            contour_detection,
            fps,
        ) = run_standard.standard_run(x1, x2, y1, y2, sub_ch, pyvips=True)
    else:
        run_standard = standard(file_name)
        (
            blur_bgsub,
            median_blur,
            bgsub,
            thresh,
            contour_detection,
            fps,
        ) = run_standard.standard_run(x1, x2, y1, y2, sub_ch)

    print("Augmentation time:", np.mean(list(blur_bgsub.values())))
    print("Detection time:", np.mean(list(contour_detection.values())))

    print("Background subtract time:", np.mean(bgsub))
    print("Median Blur time:", np.mean(median_blur))
    print("Threshold time:", np.mean(thresh))
    # set an array of sub channel dimension

    # stop the timer and display FPS information
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    print("CUDA Run")
    main(cuda=True)
    print("Standard Run")
    main(cuda=False)
    print("Pyvips Run")
    main(pyvips=True)
