import PySpin
from PyQt5.QtCore import Qt
# from PyQt5 import QtGui  # (the example applies equally well to PySide)
from PyQt5.QtWidgets import *
import cv2
from pathlib import Path
import numpy as np
import os
import math

win_width = 1600
win_height = 900

class TestSetup:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.label = QLabel()
        self.curfile = os.getcwd()
        self.rot_width = 0
        self.rot_height = 0

    def get_camera(self):
        system = PySpin.System.GetInstance()

        try:
            # [GET] camera list
            cam_list = system.GetCameras()
            cam = cam_list.GetByIndex(0)
        except:
            print("Cameras not found")
            return

        cam.Init()
        num_cameras = cam_list.GetSize()
        print("Camera Detected")
        # [LOAD] default configuration
        cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        cam.UserSetLoad()
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # turn off auto exposure
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        # Ensure desired exposure time does not exceed the maximum
        exposure_time_to_set = 20000  # in us
        exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
        cam.ExposureTime.SetValue(exposure_time_to_set)
        cam.OffsetX.SetValue(0)
        ##cam.Width.SetValue(cam.Width.GetMax())

        # set height to value
        ##cam.Height.SetValue(cam.Height.GetMax())
        cam.OffsetY.SetValue(0)
        ##cam.Height.SetValue(cam.Height.GetMax())

        # set gain
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        cam.Gain.SetValue(0)

        # frame rate
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(25)
        framerate = cam.AcquisitionFrameRate.GetValue()
        return cam

    def get_aspect(self, cam):
        cam.BeginAcquisition()
        roi_image = cam.GetNextImage()
        roi_image = np.array(roi_image.GetNDArray())
        sel = cv2.selectROI(roi_image)
        roi_x, roi_y, roi_width, roi_height = sel
        cv2.destroyAllWindows()
        cam.EndAcquisition()
        print("ROI vairables:\n x %i\n y %i\n width %i\n height %i\n" % (roi_x, roi_y, roi_width, roi_height))
        ## determine Capture X parameters
        if roi_x <= 50:
            capture_x = 0
            if roi_width + roi_x <= 1340:
                capture_width = roi_width + 50 + roi_x
            else:
                capture_width = 1440
        elif roi_x + roi_width >= 1340:
            capture_x = roi_x - 50
            capture_width = (1440 - capture_x)
        elif roi_width >= 1340:
            capture_x = 0
            capture_width = 1440
        else:
            capture_x = roi_x - 50
            capture_width = roi_width + 100

        # determine capture Y parameters
        if roi_y <= 80:
            capture_y = 0
            if roi_height + roi_y <= 920:
                capture_height = roi_height + 80 + roi_y
            else:
                capture_height = 1080
        elif roi_y + roi_height >= 920:
            capture_y = roi_y - 80
            capture_height = (1080 - capture_y)
        elif roi_height >= 920:
            capture_y = 0
            capture_height = 1080
        else:
            capture_y = roi_y - 80
            capture_height = roi_height + 160
        return (capture_x,capture_y,capture_height,capture_width) , (roi_x, roi_y, roi_width, roi_height)

    def video_capture(self):
        camera = self.get_camera()
        capture_dims, roi_dims = self.get_aspect(camera)
        roi_x, roi_y, roi_width, roi_height = roi_dims
        self.curfile = "test_avi"

        count = 0
        view_mode = 1
        save_mode = 0
        ch = 12
        sub_ch = []
        sum_ch1 = [0] * ch
        camera.BeginAcquisition()
        camera.Init()
        image_primary = camera.GetNextImage()
        frame = np.array(image_primary.GetNDArray())
        frame_crop = frame[rot_y:(rot_y + rot_height), rot_x:(rot_x + rot_width)]  # img[y:y+h, x:x+w]
        ## assigning subchannels - sub_ch
        for x in range(ch + 1):
            sub_ch_x = round(x * (roi_width / (ch)))
            sub_ch.append(sub_ch_x)


        camera.EndAcquisition()
        return None

    def save_npy(self):
        return

    def save_avi(self):
        out = cv2.VideoWriter(str(Path(str(self.curfile) + ".avi")), 0x00000000, 25.0, (self.rot_width, self.rot_height),
                                  isColor=False)
        return

    def widget_setup(self):
        '''
        Set up the GUI with only capture and save buttons.
        We will try saving the file first as a npy file then a AVI file? And benchmark both.
        '''

        app = QApplication([])
        w = QWidget()
        w.setWindowTitle('The window for testing')
        w.setMinimumWidth(win_width)
        w.setMinimumHeight(win_height)


        save_btn = ("QPushButton { background-color: rgb(25,120,250); font: bold 12px;}"
                    "QPushButton { border-color: rgb(40,40,15); border-width: 2px; }"
                    "QPushButton { border-style: inset; border-radius: 4px; padding: 10px; }"
                    "QPushButton:pressed { background-color: rgb(25,120,200) }")

        cam_btn = ("QPushButton { background-color: rgb(200,200,200); font: bold 12px;}"
                   "QPushButton { border-color: rgb(40,40,15); border-width: 2px; }"
                   "QPushButton { border-style: inset; border-radius: 4px; padding: 10px; }"
                   "QPushButton:pressed { background-color: rgb(180,180,180) }")

        save_avi_btn = ("QPushButton { background-color: rgb(25,120,250); font: bold 12px;}"
                    "QPushButton { border-color: rgb(40,40,15); border-width: 2px; }"
                    "QPushButton { border-style: inset; border-radius: 4px; padding: 10px;}"
                    "QPushButton:pressed { background-color: rgb(25,120,200) }")

        btn1 = QPushButton('Video'); btn1.setStyleSheet(cam_btn)
        btn2 = QPushButton('SAVE NPY');btn2.setStyleSheet(save_btn);btn2.setFixedSize(80, 40)
        btn3 = QPushButton('SAVE AVI'); btn3.setStyleSheet(save_avi_btn);btn3.setFixedSize(80, 40)
        btn1.setFixedSize(80, 40)

        # btn1.clicked.connect(self.video)
        # btn2.clicked.connect(self.save_data)
        # btn3.clicked.connect(self.save_avi)

        self.label.setText('Test Window')
        self.label.setAlignment(Qt.AlignCenter)

        windowLayout = QGridLayout()  ##define layout - Box type
        windowLayout.addWidget(btn3, 0, 5, 1, 1)
        windowLayout.addWidget(btn2, 0, 3, 1, 1)
        windowLayout.addWidget(btn1, 0, 1, 1, 1)

        box1 = QGroupBox('Buttons')
        box1.setFixedSize(self.width * 0.985, self.height * 0.25)
        box2 = QGroupBox('View Video')
        # box2.setFixedSize(win_height * 1.5, win_width * 0.25)
        box1.setLayout(windowLayout)

        bigLayout = QGridLayout()
        bigLayout.addWidget(box1, 2,0,1,5)
        videoLayout = QGridLayout()  ##define layout - Box type
        videoLayout.addWidget(self.label, 0, 0)
        box2.setLayout(videoLayout)
        bigLayout.addWidget(box2, 0, 0, 1, 5)
        w.setLayout(bigLayout)

        w.show()
        app.exec_()

def main():
    widget = TestSetup(win_width,win_height)
    widget.widget_setup()


if __name__=="__main__":
    main()