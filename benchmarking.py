import math
import time

import cv2
import matplotlib.pyplot as plt
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
blur_value = 7  # value = 3,5 or 7 (ODD).median Blur value determines the accuracy of detection
Delay = 1000  # time value in miliseconds. (Fastest) Minimum = 1ms
Show = 1  # To display the image. 1 = On, 0 = Off
Skip_frames = 20  # number of frames to skip before Im showing
file_name = "Raw Video Output 10x Inv-L.avi"  # Getting all open files location
Channels = 25
line_color = (200, 100, 100)


### ----- Parameters to Change ----- ###

def bgSubtract(mask, frame):
    fgmask = mask.apply(frame)
    return fgmask

'''
We test with the one avi file to benchmark
'''

def get_roi(image, roi_arr):
    print('Please select the region of interest (ROI)\n')
    sel = cv2.selectROI(image)
    roi_arr.append(list(sel))
    cv2.destroyAllWindows()

    return image, roi_arr


# def crop(frame,roi):



def to_crop(frame, r, Channels):
    ch = Channels
    #x,y represents the coordinates of the upper most corner of the rectangle
    print('[ROI] (x , y, width, height) is', r)

    # Crop image
    y1 = int(r[1])  # y
    y2 = int(r[1] + r[3])  # y + height = height of cropped
    x1 = int(r[0])  # x
    x2 = int(r[0] + r[2])  # x + width = width of cropped

    print(x1, x2, y1, y2)
    imCrop = frame[y1:(y1 + H), x1:x2]
    print(frame.shape)
    # the array for sub-channels
    sub_ch = []

    # draw lines on crop frame
    for x in range(ch + 1):
        sub_ch_x = round(x * (r[2] / (ch))) #place where line will be drawn, proportional to width
        sub_ch.append(sub_ch_x)
    #     cv2.line(imCrop, (sub_ch[x], 0), (sub_ch[x], H), line_color, 1)
    return imCrop, x1, x2, y1, y2, sub_ch

def bench_plot(blur_bgsub, contour_detection):
    aug_list = list(blur_bgsub.values())
    df_augment:DataFrame = DataFrame(data=aug_list)
    plot1 = df_augment.plot()
    plot1.set_xlabel('frames')
    plot1.set_ylabel('Time in ms')

    plt.savefig("df_augment")

    detection_list = list(contour_detection.values())
    df_detect = DataFrame(data=detection_list)
    plot2 = df_detect.plot()
    plot2.set_xlabel('frames')
    plot2.set_ylabel('Time in ms')
    plt.savefig("df_detect")



    avg_augment = np.mean(df_augment)
    avg_detect = np.mean(df_detect)

    print("Average time for bgsubtract and blur: %f\n" % avg_augment)
    print("Average time for count: %f\n" % avg_detect)

def npy_detect(npy_file, channels=25):
    error = 0
    r = [0, 0, 934, 239]
    video_frames = np.load(npy_file)
    sub_ch = [round(x * (r[2] / (channels))) for x in range(channels + 1)]
    y1 = int(r[1])  # y
    y2 = int(r[1] + r[3])  # y + height = height of cropped
    x1 = int(r[0])  # x
    x2 = int(r[0] + r[2])  # x + width = width of cropped

    for frame in video_frames:
        pic = frame[y1:(y1 + H), x1:x2]
        # crop = bgSubtract(mask,pic)

        mask = cv2.createBackgroundSubtractorMOG2(history=3,
                                                  varThreshold=100,
                                                  detectShadows=False)
        crop = mask.apply(pic)
        crop = cv2.medianBlur(crop, blur_value)
        crop = cv2.threshold(crop, 125, 255, cv2.THRESH_BINARY)[1]
        image, contours, hierarchy = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # list of all the coordinates (tuples) of each cell
        coord_list = []

        # to find the coordinates of the cells
        sum_ch1 = []
        for i in range(len(contours)):
            avg = np.mean(contours[i], axis=0)
            coord = (int(avg[0][0]), int(avg[0][1]))  ##Coord is (y,x)
            if Show == 1:
                cv2.circle(pic, coord, 10, (255, 0, 255), 1)
            ch_pos = int(math.floor((coord[0]) / sub_ch[1]))
            try:
                sum_ch1[ch_pos] += 1
            except:
                error += 1

        return sum_ch1

def save_excel(sum_ch1):
    total_sum = []
    total_sum.append(sum_ch1)

    ###write dataframes and export to an Excel file
    check = 0
    title = []
    for j in range(len(total_sum)):
        if check < len(total_sum[j]): check = len(total_sum[j])
        title.append('Run 1')

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

def main(test_npy=False):
    # Get ROI from frames
    stream = cv2.cuda_Stream()
    cap = FileVideoStream(file_name).start()
    image = cap.read()
    print('***** PROCESSING ROI for RUN 1 ***** File: %s' % file_name)
    cap.stop()
    '''
    Get crop size and draw lines
    '''
    print('***** PROCESSING RUN 1 ***** File: %s' % file_name)
    # # Read image start image
    # ret, frame = cap.read()

    r = [0, 0, 934, 239]
    prep_crop, x1, x2, y1, y2, sub_ch = to_crop(image, r, Channels)
    #
    # cv2.namedWindow('Cropped Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Cropped Image', prep_crop)
    # cv2.waitKey(Delay)

    '''
    Background Subtract
    '''
    threshold_mat = cv2.cuda_GpuMat()
    count = 0
    spot_all = []
    mask:cv2.cuda.createBackgroundSubtractorMOG2 = cv2.cuda.createBackgroundSubtractorMOG2(history=3,
                                              varThreshold=100,
                                              detectShadows=False)
    filter = cv2.cuda.createMedianFilter(cv2.CV_8UC1, blur_value)
    sum_ch1 = np.zeros(28)
    #
    # # metrics
    fps = FPS().start()
    start = time.time()
    error = 0
    contour_detection = {}
    blur_bgsub = {}
    median_blur = []
    bgsub = []
    thresh = []
    # run count
    gpu_frames = []
    gpu_read = cvs.video_queue(file_name,y1,H,x1,x2).start()
    while gpu_read.more():
        #load frames to memory
        count += 1
        frame = gpu_read.read()
        # print(count)
    gpu_read.stop()
    gpu_read = cvs.video_queue(file_name, y1, H, x1, x2).start()
    while gpu_read.more():
        cycle_start = time.time()
        frame = gpu_read.read()
        augment_start = time.time()
        # crop = bgSubtract(mask,pic)
        blur = time.time()
        filter.apply(frame)
        blur_stop = time.time()
        median_blur.append(blur_stop - blur)

        bg = time.time()
        aug_frame = mask.apply(frame, -1, stream)
        bg_stop = time.time()
        bgsub.append(bg_stop - bg)
        crop = aug_frame.download()
        threshh = time.time()
        crop = cv2.threshold(crop, 125, 255, cv2.THRESH_BINARY)[1]
        thresh_stop = time.time()
        thresh.append(thresh_stop-threshh)
        augment_end = time.time()
        blur_bgsub[count] = augment_end*1000.0 - augment_start*1000.0

        '''
        Contour Detection
        '''

        count_start = time.time()
        contours, hierarchy = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # list of all the coordinates (tuples) of each cell
        coord_list = []
        # print(contours)
        # to find the coordinates of the cells
        for i in range(len(contours)):
            avg = np.mean(contours[i], axis=0)
            coord = (int(avg[0][0]), int(avg[0][1]))  ##Coord is (y,x)
            ch_pos = int(math.floor((coord[0]) / sub_ch[1]))
            try:
                sum_ch1[ch_pos] += 1
            except:
                error += 1
        count_end = time.time()
        contour_detection[count] = count_end*1000.0 - count_start*1000.0

        fps.update()
        cycle_end = time.time()

    end = time.time()
    fps.stop()
    detect_benchmark = end - start
    print("Time taken for counting:",detect_benchmark)

    # print("contour detection:",contour_detection)
    # print("augment:", blur_bgsub)
    # cv2.destroyAllWindows()
    # bench_plot(blur_bgsub, contour_detection)
    print("Augmentation time:", np.mean(list(blur_bgsub.values())))
    print("Detection time:", np.mean(list(contour_detection.values())))

    print("Background subtract time:",np.mean(bgsub))
    print("Median Blur subtract time:", np.mean(median_blur))
    print("Threshold subtract time:", np.mean(thresh))
    # set an array of sub channel dimension
    print('[RESULTS] for RUN is ', sum_ch1)
    print('[ERROR] Count is: ', error)

    # stop the timer and display FPS information
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print('[INFO] Each cycle time taken = %0.5fs' % (cycle_end - cycle_start))
    print('----------------------------------------------------------------------')

    cv2.destroyAllWindows()

    if test_npy:
        npy_frames = np.load("test_file.npy")
        sum_ch1 = npy_detect(npy_frames)

    save_excel(sum_ch1)
    plt.show()


if __name__ == "__main__":
    main()

