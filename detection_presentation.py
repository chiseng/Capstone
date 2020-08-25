import csv
import math
import queue
import time
import multiprocessing
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import FPS
from pandas import DataFrame
from file_conversion import *
from pathlib import *


# file_name = "WBC285 inv-L-pillars -350mbar 150fps v3.4.avi"


def to_crop(frame, r, Channels):
    ch = Channels

    print("[ROI] (x , y, width, height) is", r)

    offset = 3  # We do not take the first 2 channels because only a minority will flow through and will be RBC
    # Crop image
    y1 = int(r[1])  # y
    y2 = int(r[1] + r[3])  # y + height = height of cropped
    x1 = int(r[0])  # x
    x2 = int(r[0] + r[2])  # x + width = width of cropped

    print(x1, x2, y1, y2)
    imCrop = frame[y1:(y2), x1:x2]
    print(frame.shape)
    # the array for sub-channels
    sub_ch = []
    ch_length = r[2] / ch
    # draw lines on crop frame
    for x in range(offset, ch + offset + 1):
        sub_ch_x = round(
            x * (r[2] / ch_length)
        )  # place where line will be drawn, proportional to width
        sub_ch.append(sub_ch_x)

    return x1, x2, y1, y2, sub_ch, ch_length


def save_excel(sum_ch1):
    sum_ch1 = np.load("run_results_1.npy")
    channels = 35  # preset channel size
    title = []
    total_sum = [item for item in sum_ch1]
    for i in range(len(total_sum), channels):
        total_sum.append(0)


    with open("test.csv", 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(total_sum)
    

class standard:
    def __init__(self, filename, Channels):
        self.mask: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=3, varThreshold=190, detectShadows=False
        )
        self.mask2: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )
        self.sum_ch1 = np.zeros(34)
        self.frames_buffer = []
        self.rbc_counting = np.zeros(Channels)
        self.cycle_count = 0
        self.filename = filename
        self.video_buffer = []
        self.queue = multiprocessing.JoinableQueue()

    def plot_fig(self):
        sum_ch1 = np.load("run_results.npy")
        channels = [i for i in range(len(sum_ch1))]
        plt.bar(channels, sum_ch1)
        plt.plot(sum_ch1, color="black")
        plt.xlabel("Channel")
        plt.ylabel("Cell Count")
        plt.show()
    
    def image_aug(self, frame):

        frame = self.unsharp_mask(frame, sigma=2.0)  # increase sharpness
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))  # merge channels
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    def unsharp_mask(
        self, image, kernel_size=(7, 7), sigma=1.0, amount=1.0, threshold=0
    ):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def test_augmentation(self, image):
        ret = cv2.GaussianBlur(image, (5, 5), 2)
        # ret = cv2.bilateralFilter(image, 10, 150,150)
        _, ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ret

    def rbc_channels(self, sum_ch1):
        threshold = 0.001 * np.sum(sum_ch1)
        mx = np.max(sum_ch1)
        # mask = (sum_ch1 >= threshold)
        return sum_ch1[: np.where(sum_ch1 == mx)[0][0] + 3]

    def rbc_detection(self) -> np.ndarray:
        for item in self.frames_buffer:
            roi = [84, 325, 1095, 160]
            frame = item[roi[1] : (roi[1] + roi[3]), roi[0] : (roi[0] + roi[2])]
            Channels = 34
            frame = self.image_aug(frame)

            frame = self.mask2.apply(frame)
            frame = self.test_augmentation(frame)

            channel_len = roi[2] / Channels
            contours, hierarchy = cv2.findContours(
                frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for i in range(len(contours)):
                avg = np.mean(contours[i], axis=0)
                coord = (int(avg[0][0]), int(avg[0][1]))  # Coord is (x,y)
                ch_pos = int(math.floor((coord[0]) / channel_len))

                try:
                    self.rbc_counting[ch_pos] += float(1)
                except:
                    print("Array error")
                    break
        return self.rbc_counting

    
    def standard_run(
        self,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        sub_ch: list,
        channel_len,
    ):

        fps = FPS().start()
        cap = FileVideoStream(self.filename).start()
        count = 0


        start = time.time()
        cycle_start = time.time()
        frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        while cap.more():
#            print(count)
            frame = cap.read()
            if frame is None:
                break

            # if count < 200:
            #     self.frames_buffer.append(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame[y1:y2, x1:x2]
            if not Path("/home/smart/output_frames").is_dir():
                Path.mkdir(Path("/home/smart/output_frames"))
            cv2.imwrite("/home/smart/output_frames/img" + str(count) + ".png", frame)
            # crop = self.mask.apply(frame)
            # crop = cv2.GaussianBlur(crop, (7, 7), 3.0)
            #
            # _, crop = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # contours, hierarchy = cv2.findContours(
            #     crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            # )
            #
            # for i in range(len(contours)):
            #     avg = np.mean(contours[i], axis=0)
            #     coord = (int(avg[0][0]), int(avg[0][1]))  # Coord is (x,y)
            #     ch_pos = int(math.floor((coord[0]) / channel_len))
            #
            #
            #     try:
            #         self.sum_ch1[ch_pos] += float(1)
            #     except:
            #         pass
            #
            count += 1
            fps.update()
        fps.stop()
        cycle_end = time.time()
        self.cycle_count += 1
        end = time.time()
        detect_benchmark = end - start
        print("Number of frames processed: ", count)
        print("Time taken for WBC counting:", detect_benchmark)
        print(
            "[INFO] Each cycle time taken = %0.5fs"
            % ((cycle_end - cycle_start) / count)
        )

        return fps

    """
    RBC counts: [1.236e+03 1.768e+03 8.640e+02 8.000e+00 2.000e+00 1.000e+00 2.000e+00
 0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
 1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00]
    """

    def process_results(self, rbc_counting):
        rbc_roi = self.rbc_channels(rbc_counting)
        wbc_roi = self.sum_ch1[len(rbc_roi) :]

        total_count = np.concatenate((rbc_roi, wbc_roi))
        save_excel(total_count)
        np.save(f"run_results_{self.cycle_count}", total_count)
        print(f"[ROI] for WBC: {wbc_roi}")
        print(f"[ROI] for RBC: {rbc_roi}")
        print(f"RBC counts: {rbc_counting}")
        print(f"[RESULTS] for RUN is {total_count}")


def main():
    # Get ROI from frames
    file_name = "/home/smart/WBC286 InvL-Pillars -35mbar 15fps 29-11-2019 v3.4.avi"
    #file_name = "C:/Users/Me/Desktop/capstone/WBC286 InvL-Pillars -350mbar 150fps 29-11-2019 v3.4.avi"
    cap = FileVideoStream(file_name).start()
    image = cap.read()
    print("***** PROCESSING ROI for RUN 1 ***** File: %s" % file_name)
    cap.stop()
    print("***** PROCESSING RUN 1 ***** File: %s" % file_name)
    print("Frame size: ", image.shape)
    Channels = 34
    #original size: [502,1402,3]
    # r = [84, 357, 1238, 130]
    r = [int(0.167 * image.shape[0]), int(0.25 * image.shape[1]), int(0.883 * image.shape[1]), 130]
    x1, x2, y1, y2, sub_ch, channel_len = to_crop(image, r, Channels)

    # run count
    run_standard = standard(file_name, Channels)
    (fps) = run_standard.standard_run(x1, x2, y1, y2, sub_ch, channel_len)
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    rbc_counts = run_standard.rbc_detection()
    run_standard.process_results(rbc_counts)

    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    print("Standard Run")
    main()
