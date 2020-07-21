import math
# import pyvips
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import FPS
from imutils.video import FileVideoStream
from pandas import DataFrame

blur_value = 7
#file_name = "WBC285 inv-L-pillars -350mbar 150fps v3.4.avi"
file_name = "/home/smart/WBC286 InvL-Pillars -35mbar 15fps 29-11-2019 v3.4.avi"
Channels = 34
offset = 3  # We do not take the first 2 channels because only a minority will flow through and will be RBC
line_color = (200, 100, 100)


"""
We test with the one avi file to benchmark
"""

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
    #     cv2.line(imCrop, (sub_ch[x], 0), (sub_ch[x], H), line_color, 1)

    return x1, x2, y1, y2, sub_ch, ch_length


def save_excel(sum_ch1):
    total_sum = []
    total_sum.append(sum_ch1)
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
    df = DataFrame(data=TTotal_sum, columns=title)
    df.to_excel("testfile" + ".xlsx", index=False, sheet_name="Results")


class standard:
    def __init__(self, filename):
        self.mask: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=3, varThreshold=190, detectShadows=False
        )
        self.mask2: cv2.createBackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=3, varThreshold=100, detectShadows=False
        )
        self.sum_ch1 = np.zeros(34)
        # self.blur_bgsub = {}
        # self.contour_detection = {}
        # self.median_blur = []
        # self.bgsub = []
        # self.thresh = []
        self.frames_buffer = []
        self.rbc_counting = np.zeros(Channels)
        self.cycle_count = 0

    def vips_to_array(self, vi):
        return np.ndarray(
            buffer=vi.write_to_memory(),
            dtype=format_to_dtype[vi.format],
            shape=[vi.height, vi.width, vi.bands],
        )

    def vips_filter(self, frame):
        height, width = frame.shape
        linear = frame.reshape(width * height)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, 1, "uchar")
        filtered = vi.gaussblur(0.7)
        image = self.vips_to_array(filtered)
        return image

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

        # cv2.imshow("Sharpened", sharpened_frame)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

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
            # if count == 4:
            #     long_imshow(frame)
            # bg_sub_frame = mask.apply(framee)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            # ret_frame = test_augmentation(bg_sub_frame)

            channel_len = roi[2] / Channels
            contours, hierarchy = cv2.findContours(
                frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for i in range(len(contours)):
                avg = np.mean(contours[i], axis=0)
                coord = (int(avg[0][0]), int(avg[0][1]))  # Coord is (x,y)
                ch_pos = int(math.floor((coord[0]) / channel_len))
                # cv2.circle(frame, coord, 10, (255, 0, 255), 1)
                # cv2.imshow(f"frame", frame)
                # cv2.waitKey(500)
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
        pyvips=False,
    ):
        fps = FPS().start()
        cap = FileVideoStream(file_name).start()
        count = 0

        # root = pathlib.Path("out_frames")
        # if not root.exists():
        #     root.mkdir()
        start = time.time()
        cycle_start = time.time()
        while cap.more():
            frame = cap.read()
            if frame is None:
                break

            if count < 200:
                self.frames_buffer.append(frame)
            frame = frame[y1:y2, x1:x2]
            augment_start = time.time()
            # crop = bgSubtract(mask,pic)
            # bg = time.time()
            crop = self.mask.apply(frame)
            # bg_stop = time.time()
            # self.bgsub.append(bg_stop - bg)
            # blur = time.time()
            if pyvips:
                crop = self.vips_filter(crop)
            else:
                crop = cv2.GaussianBlur(crop, (7, 7), 3.0)

            # blur_stop = time.time()
            # self.median_blur.append(blur_stop - blur)

            # threshh = time.time()
            _, crop = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # thresh_stop = time.time()
            # self.thresh.append(thresh_stop - threshh)
            # augment_end = time.time()
            # self.blur_bgsub[self.count] = augment_end - augment_start
            # self.count += 1

            # count_start = time.time()
            contours, hierarchy = cv2.findContours(
                crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # list of all the coordinates (tuples) of each cell
            # print(contours)
            # to find the coordinates of the cells
            for i in range(len(contours)):
                avg = np.mean(contours[i], axis=0)
                coord = (int(avg[0][0]), int(avg[0][1]))  # Coord is (x,y)
                ch_pos = int(math.floor((coord[0]) / channel_len))

                try:
                    self.sum_ch1[ch_pos] += float(1)
                except:
                    pass
            # cv2.imshow(f"frame {count}", crop)
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()
            count += 1
            # count_end = time.time()
            # self.contour_detection[self.count] = count_end - count_start
            fps.update()
        cycle_end = time.time()
        self.cycle_count += 1
        end = time.time()
        fps.stop()
        detect_benchmark = end - start
        print("Number of frames processed: ", count)
        print("Time taken for WBC counting:", detect_benchmark)
        print("[INFO] Each cycle time taken = %0.5fs" % ((cycle_end - cycle_start)/count))
        return (
            # self.blur_bgsub,
            # self.median_blur,
            # self.bgsub,
            # self.thresh,
            # self.contour_detection,
            fps
        )

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
        np.save(f"run_results_{self.cycle_count}", total_count)
        print(f"[ROI] for WBC: {wbc_roi}")
        print(f"[ROI] for RBC: {rbc_roi}")
        print(f"RBC counts: {rbc_counting}")
        print(f"[RESULTS] for RUN is {total_count}")


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

    r = [84, 357, 1238, 130]
    x1, x2, y1, y2, sub_ch, channel_len = to_crop(image, r, Channels)

    """
    Background Subtract
    """

    # run count
    run_standard = standard(file_name)
    (
        # blur_bgsub,
        # median_blur,
        # bgsub,
        # thresh,
        # contour_detection,
        fps
    ) = run_standard.standard_run(x1, x2, y1, y2, sub_ch, channel_len)
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    rbc_counts = run_standard.rbc_detection()
    run_standard.process_results(rbc_counts)

    # print("Augmentation time:", np.mean(list(blur_bgsub.values())))
    # print("Detection time:", np.mean(list(contour_detection.values())))
    #
    # print("Background subtract time:", np.mean(bgsub))
    # print("Median Blur time:", np.mean(median_blur))
    # print("Threshold time:", np.mean(thresh))
    # set an array of sub channel dimension


    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    # print("CUDA Run")
    # main(cuda=True)
    print("Standard Run")
    main(cuda=False)
    # print("Pyvips Run")
    # main(pyvips=True)


"""
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
    elif pyvips:
        run_standard = standard(file_name)
        (
            blur_bgsub,
            median_blur,
            bgsub,
            thresh,
            contour_detection,
            fps,
        ) = run_standard.standard_run(x1, x2, y1, y2, sub_ch, channel_len, pyvips=True)
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
        self.sum_ch1 = np.zeros(Channels)
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
            crop = cv2.threshold(crop, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
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
"""
