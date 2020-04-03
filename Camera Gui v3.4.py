from PyQt5 import QtGui  # (the example applies equally well to PySide)
from PyQt5.QtGui import QImage, QPixmap, QPicture, QTransform
from PyQt5.QtWidgets import QCheckBox, QMessageBox, QTabWidget, QComboBox
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtWidgets import QGroupBox, QDialog, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QListWidget, QProgressBar
from PyQt5.QtWidgets import QListWidget, QSlider
from PyQt5.QtCore import Qt
import PySpin
import pyqtgraph as pg
import pyqtgraph as ptime
import cv2
import numpy as np
import time
import math
from datetime import datetime
import os
from pathlib import Path
from pandas import DataFrame
import csv

### ----- Parameters to Change ----- ###
win_length = 1375  #Divisible by 6 as the ratio of panel to view window is 1:5
win_width = 750
win_scale = False
no_cancel = False
cwDir = 0
file_stem = 'Name Your File'
filename = 'file.save'
console_log = []
console_num = 0
filelist = [0]
got_camera = False
roi_mode = 0
view_mode = 0
save_mode = 0


roi_x, roi_y , roi_width, roi_height = (0, 0, 0, 0)
capture_x,capture_y, capture_width, capture_height = (0, 0, 0, 0)

exposure_time = 100       #micro seconds min=10us (FIXED)
FPS = 25                    #Frame rate to capture 
gain = 0                   #Sensor gain (FIXED)
image_format = 'Mono8'      #image pixel format. 'Mono8' or 'Bayer' for BGR
width = 1440               #image width in pixels (FIXED)
height = 600                #image height in pixels (FIXED)
Show = 1                    #Show = show the video while loop
Skip_frames = 25            #number of frames to skip before imshow
num_frames = 1000            #number of frames to view
video_record = 0            #1: to record the video, 0: No Record
video_mode = 'MJPG'          #AVI or MJPG 
video_name = 'zdddd.avi'    #Filename to save as

#angle variables
angle = 0                   #rotational angle
rot_x = 0                   #x pos value of final rotated and cropped image
rot_y = 0                   #y pos value of final rotated and cropped image
rot_width = 0               #width of final rotated and cropped image
rot_height = 0              #height of final rotated and cropped image


font = cv2.FONT_HERSHEY_SIMPLEX

H = 110             #No. of pixels to select for height of Region Of Interest
blur_value = 3      #value = 3,5 or 7 (ODD).median Blur value determines the accuracy of detection
Delay = 1           #time value in miliseconds. (Fastest) Minimum = 1ms
Show = 1            #To display the image. 1 = On, 0 = Off
Skip_frames = 10     #number of frames to skip before Im showing
Channels = 35       #number of Channels
Export_Results = 1  #export data to excel

### ----- Initialisation ----- ###
mask = cv2.createBackgroundSubtractorMOG2(history = 3,
                                          varThreshold = 100,
                                          detectShadows = False)
sum_ch1 = [0]*Channels
cwDir = os.getcwd()

'''
Global parameters used, making classes or setter/getter would be better
'''
### ----- Parameters to Change ----- ###
def reset_parameters():
    global FPS, exposure_time, num_frames, gain, width, height
    FPS = 25
    exposure_time = 100000000
    num_frames = 250
    gain = 0
    width = 1440
    height = 1080

    cam_field2.setText(str(FPS))
    cam_field3.setText(str(exposure_time))
    cam_field4.setText(str(num_frames))
    cam_field5.setText(str(gain))
    cam_field6.setText(str(width))
    cam_field7.setText(str(height))
    return

def cancel():
    global no_cancel
    if no_cancel == True:
        no_cancel = False
        console_print('CANCEL button pressed')
        camera_reset()
    else: return

def rotation_value():
    global angle
    angle = (slide1.value()/5)
    console_print('The rotation angle is '+str(angle))
    return angle

def showdialog():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("Saved Parameters")
    msg.setInformativeText("H = %i, Ch = %i, Blur = %i, Delay = %ims, Skip_frames = %i. Show = %i"%(H,Channels, blur_value,Delay,Skip_frames,Show))
    msg.setWindowTitle("Saving Parameters")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()
    return

def select_folder():
    global cwDir
    cwd = os.getcwd()
    cwDir = QtGui.QFileDialog.getExistingDirectory(None, 
                                                'Open working directory', 
                                                cwd, 
                                                QtGui.QFileDialog.ShowDirsOnly)
    cam_field1.setText(cwDir)

def select_files():
    global filelist
    options = QtGui.QFileDialog.Options()
    options |= QtGui.QFileDialog.DontUseNativeDialog
    
    filelist, suff = QtGui.QFileDialog.getOpenFileNames(None,
                                                        "LALALALALAL",
                                                        "",
                                                        "video (*.avi)", #"video (*.avi);;All Files (*)"
                                                        options=options)
    if not(filelist):
        console_print('Error Loading | No files selected')
        field1.setText('No File(s) Selected')
        print(filelist)
        return
    
    file = Path(filelist[0])
    field1.setText(str(file))
    if len(filelist) == 1:
        console_print('Selected: ' + str(filelist[0]))
    elif len(filelist) > 1:
        console_print('Selected: ' + str(len(filelist)) + ' Files')

    return
# prints DMY on console
##########################################
# IMAGE CAPTURING
##########################################

def camera_start():
    global got_camera, cam, cam_list

    if got_camera == True: return
    ## --- Setup Camera --- ##
    # [GET] system
    system = PySpin.System.GetInstance()

    try:
        # [GET] camera list
        cam_list = system.GetCameras()
        # Init_camera
        cam = cam_list.GetByIndex(0)
    except:
        got_camera = False
        return

    # [Initialize] camera
    cam.Init()
    cam.BeginAcquisition()
    cam.EndAcquisition()
    num_cameras = cam_list.GetSize()
    print('[INFO] Number of cameras detected: %d' % num_cameras)
    # [LOAD] default configuration
    cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
    cam.UserSetLoad()
    got_camera = True

    return

def camera_reset():
    cam.OffsetY.SetValue(0)
    cam.OffsetX.SetValue(0)
##    max_width = cam.Width.GetMax()
##    max_height = cam.Height.GetMax()
##    cam.Width.SetValue(max_width)
##    cam.Height.SetValue(max_height)
    return

def camera_properties(c_width, c_height, c_fps):
    global exposure_time, width, height
    width = c_width
    height = c_height
    # [SET] Image dimenstion width and height and center ROI
    max_width = cam.Width.GetMax()
    max_height = cam.Height.GetMax()
    if width <= max_width and width >= 100:
        if width%4 != 0:
            print(width, 'not divisible by 4')
            width=width-width%4
    elif width < 100 and width > 0:
        width = 100
    else: width = max_width
    cam_field6.setText(str(width))
    
    if height <= max_height and height >= 100:
        if height%2 != 0:
            print(height, 'not divisible by 2')
            height=height+1
    elif height < 100 and height > 0:
        height = 100
    else: height = max_height
    cam_field7.setText(str(height))
    cam.Width.SetValue(width)
    cam.Height.SetValue(height)
    cam.OffsetY.SetValue(round((max_height-height)/2)-round((max_height-height)/2)%2)
    cam.OffsetX.SetValue(round((max_width-width)/2)-round((max_width-width)/2)%4)
    print('[INFO] Frame Width =',width,'| Height =',height)

    # [SET] analog. Set Gain. Turn off Gamma.
    gain = cam_field5.text()
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(int(gain))
    cam.GammaEnable.SetValue(False)
    print('Cam Properties gain End')
    
    # [SET] acquisition. Continues acquisition. Auto exposure off. Set frame rate. 
    exposure_time = cam_field3.text()
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    exposure_time = min(cam.ExposureTime.GetMax(), int(exposure_time))
    cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
    cam.ExposureTime.SetValue(int(exposure_time))
    print('Cam Properties exposure tim End')

    # [SET] Framerates
    cam.AcquisitionFrameRateEnable.SetValue(True)
    max_FPS = cam.AcquisitionFrameRate.GetMax()
    try: int(c_fps)
    except:    c_fps = 25
    if int(c_fps) > (round(max_FPS)-1):
        c_fps = (round(max_FPS)-1)
    elif int(c_fps) >= 1 and int(c_fps)<= (round(max_FPS)-1):
        pass
    else: c_fps = 25
    
    cam.AcquisitionFrameRate.SetValue(int(c_fps))
    print('Cam Properties End')
    return

def video():
## cv2 benchmark
    global H, Channels, blur_value, Delay, Skip_frames, Show, no_cancel, angle

    try:
        H = round(float(field2.text()))
        Channels = round(float(field3.text()))
        blur_value = round(float(field4.text()))
        Delay = round(float(field5.text()))
        Skip_frames = round(float(field6.text()))
        if chk_box1.isChecked():
            Show = 1
        else:
            Show = 0
            btn2.setEnabled(False)
    except:
        console_print('There is an error in the field values')
        H = 110             #No. of pixels to select for height of Region Of Interest
        blur_value = 3      #value = 3,5 or 7 (ODD).median Blur value determines the accuracy of detection
        Delay = 1           #time value in miliseconds. (Fastest) Minimum = 1ms
        Show = 1            #To display the image. 1 = On, 0 = Off
        Skip_frames = 10     #number of frames to skip before Im showing
        Channels = 35       #number of Channels

    if H < 10:
        H = 10
        field2.setText(str(10))
        
    if Channels < 2:
        Channels = 2
        field3.setText(str(2))

    if (blur_value%2 == 0 or blur_value < 3) or blur_value > 15:
        blur_value = 3
        field4.setText(str(3))
        
    if Delay <1:
        Delay = 1
        field5.setText(str(1))
        
    if Skip_frames <1:
        Skip_frames = 1
        field6.setText(str(1))
        
    if (not(filelist) or filelist[0] == 0):
        return console_print('No File Selected')
    
    no_cancel = True
    
    width = w.frameGeometry().width()
    height = w.frameGeometry().height()

    btn1.setEnabled(False)
    btn4.setEnabled(False)
    
    cv = 0
    name = Path(filelist[0])
    
    cap = cv2.VideoCapture(filelist[0])
    ret, frame = cap.read()
    size = frame.shape
    x1 = size[1]
    y1 = size[0]
    step = frame.size / (size[0]) / 3
    qformat = QImage.Format_Indexed8
    
    console_print('Playing File: ' + str(filelist[0]))
    bar1.setMaximum(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #init bar size
    cap = cv2.VideoCapture(filelist[0])
    start= time.time()
    
    while(cap.isOpened() and no_cancel):
        ret, frame = cap.read()
        
        if not ret: break
       #blur so that noise isnt detected.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        crop = mask.apply(frame)
        crop = cv2.medianBlur(frame,blur_value)
        crop = cv2.threshold(crop, 125, 255, cv2.THRESH_BINARY)[1]
        
        if chk_box1.isChecked() and cv%Skip_frames == 0:
            if chk_box2.isChecked():
                frame = cv2.flip(frame, -1)
            img = QImage(frame, size[1], size[0], step, qformat)
            img = QPixmap.fromImage(img)
            rotate = QTransform().rotate(angle)
            img = img.transformed(rotate, Qt.SmoothTransformation)
            rad = math.radians(abs(angle))
            cos = math.cos(rad)
            sin = math.sin(rad)
            x2 = (x1*cos) + (y1*sin)
            rot_width = x1 - abs(x2-x1)
            rot_height = (y1 * cos) - (x1 * sin)
            img = img.copy( abs(x2-x1), abs(x1*sin), int(rot_width), int(rot_height))    # copy(int x, int y, int width, int height)
            if chk_box3.isChecked():
                img=img.scaled(round(width*(5/6))-50, height-25, aspectRatioMode = Qt.KeepAspectRatio)
            label.setPixmap(img)
            cv2.waitKey(Delay) 
            
        cv += 1
        bar1.setValue(cv)
    print(abs(x2-x1), abs(x1*sin), int(rot_width), int(rot_height))
    end = time.time()
    label.setText("Hello...What's next?")
    #cv2.destroyAllWindows()
    console_print("H = %i, Ch = %i, Blur = %i, Delay = %ims, Skip_frames = %i. Show = %i"%(H,Channels, blur_value,Delay,Skip_frames,Show))
    console_print('Opencv2 Processing FPS = %0.3f'%(cv/(end-start)))
    console_print('Total number of frames = ' + str(cv))
    no_cancel = False
    btn1.setEnabled(True)
    btn2.setEnabled(True)
    btn4.setEnabled(True)
    return

def camera_view():
    global no_cancel, got_camera, frame, angle
    global view_mode, save_mode
    global rot_x, rot_y, rot_width, rot_height
    # [GET] camera list
    cam_list = system.GetCameras()
    print(cam_list)
    num_cameras = cam_list.GetSize()
    print('num cams',num_cameras)
    camera_start()
    if got_camera == True and num_cameras == 0:
        print('trouble')
        console_print('The camera was disconnected')
        got_camera = False
        return
            
    if got_camera == False:
        console_print('No Detectable Camera')
        return
    btn5.setEnabled(False)
    btn4.setEnabled(False)
    btn1.setEnabled(False)
    btn3.setEnabled(False)
    no_cancel = True
    
    console_print('Camera View BEGIN')
    print('Initial FPS view = ',cam.AcquisitionFrameRate.GetValue())
    c_width = cam_field6.text()
    c_height = cam_field7.text()
    camera_properties(int(c_width) , int(c_height), 25)
    cam.BeginAcquisition()
    count = 0
    view_mode = 1
    save_mode == 0
    local_angle = (angle+1)   #arbitury number assigned
    ch = int(field3.text())
    sub_ch = []
    
    while no_cancel:
        image_primary = cam.GetNextImage()
        frame = np.array(image_primary.GetNDArray())
        if chk_box2.isChecked():
            frame = cv2.flip(frame, -1)

        if (local_angle != angle):
            rad = math.radians(abs(angle))
            cos = math.cos(rad)
            sin = math.sin(rad)
            y1,x1 = frame.shape
            image_center = (x1 / 2, y1 / 2)
            y2 = (y1 * cos) + (x1 * sin)
            x2 = (x1 * cos) + (y1 * sin)
            rot_x = int(abs(x2-x1))
            rot_y = int(abs(x1*sin))
            rot_width = int(x1 - abs(x2-x1))
            rot_height = int((y1 * cos) - (x1 * sin))
            M = cv2.getRotationMatrix2D((image_center),angle,1)
            M[0, 2] += ((x2 / 2) - image_center[0])
            M[1, 2] += ((y2 / 2) - image_center[1])
                      
            print('angle has changed')
            local_angle = angle
        
        frame = cv2.warpAffine(frame,M,(int(x2),int(y2)))  
        frame_crop = frame[rot_y:(rot_y+rot_height) , rot_x:(rot_x+rot_width)]  #img[y:y+h, x:x+w]

        if chk_box4.isChecked() and (capture_width+capture_height) > 0:
            cv2.putText(frame_crop, 'ROI', (roi_x + 10, roi_y + 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_crop, 'Capture Size', (capture_x + 10, capture_y + 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame_crop, (roi_x,roi_y), (roi_x+roi_width, roi_y+roi_height), (255,255,255), 3,1,0)
            cv2.rectangle(frame_crop, (capture_x ,capture_y), (capture_x + capture_width, capture_y + capture_height), (255,255,255), 3,1,0)
            # draw lines on crop frame
            ## 
            for x in range(ch):
                sub_ch_x = round(x*(roi_width/(ch)))
                sub_ch.append(sub_ch_x)
                cv2.putText(frame_crop, str(x+1), ((roi_x+sub_ch[x]+10), (roi_y + int(roi_width*110/840) - 25)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(frame_crop, ((roi_x+sub_ch[x]),roi_y), ((roi_x+sub_ch[x]), (roi_y+int(roi_width*110/840))), (200,200,100),2) 

        size = frame.shape
        step = frame.size / (size[0])
        
        qformat = QImage.Format_Indexed8
##        cv2.imshow('Crop', frame_crop)
        img = QImage(frame, size[1], size[0], step, qformat)
        img = img.copy( rot_x, rot_y, rot_width, rot_height)    # copy(int x, int y, int width, int height)
        img = QPixmap.fromImage(img)
        

        # rotation

##        rotate = QTransform().rotate(angle)
##        img = img.transformed(rotate, Qt.SmoothTransformation)
##        rad = math.radians(abs(angle))
##        cos = math.cos(rad)
##        sin = math.sin(rad)
##        x2 = (x1*cos) + (y1*sin)
##        rot_width = x1 - abs(x2-x1)
##        rot_height = (y1 * cos) - (x1 * sin)
##        img = img.copy( abs(x2-x1), abs(x1*sin), int(rot_width), int(rot_height))    # copy(int x, int y, int width, int height)

        # img show
        if chk_box3.isChecked():
            img=img.scaled(round(width*(0.6))-50, round(height*(0.6))-25, aspectRatioMode = Qt.KeepAspectRatio)
        label.setPixmap(img)
        cv2.waitKey(Delay)
        image_primary.Release()

    cam.EndAcquisition()
    print('FPS view = ',cam.AcquisitionFrameRate.GetValue())
    view_mode = 0
    label.setText("Hello...What's next?")
    camera_reset()
    no_cancel = False
    
    btn1.setEnabled(True)
    btn4.setEnabled(True)
    btn3.setEnabled(True)
    btn5.setEnabled(True)

    return

def roi_select():
    global roi_mode
    global roi_x, roi_y, roi_width, roi_height
    global capture_x, capture_y, capture_width, capture_height
    if no_cancel == True:
        print('camera is on')
        cap = cv2.VideoCapture(0)
        ret,frame = cap.read()

        if chk_box2.isChecked():
            #frame_error here
            frame = cv2.flip(frame, -1)
        print(2)
##        cv2.putText(frame_crop, 'Please select a region of interest and press ENTER', (30, 30), font, 1, (2, 255, 255), 1, cv2.LINE_AA)

        sel = cv2.selectROI(frame)
        print(3)
        cv2.destroyAllWindows()
        chk_box4.setCheckState(True)
        
    else:
        print('can capture')
        camera_start()
        camera_reset()
        c_width = cam_field6.text()
        c_height = cam_field7.text()
        camera_properties(int(c_width) , int(c_height), 25)
        cam.BeginAcquisition()
        roi_image = cam.GetNextImage()
        roi_image = np.array(roi_image.GetNDArray())

    if angle != 0:
        rad = math.radians(abs(angle))
        cos = math.cos(rad)
        sin = math.sin(rad)
        y1,x1 = roi_image.shape
        image_center = (x1 / 2, y1 / 2)
        y2 = (y1 * cos) + (x1 * sin)
        x2 = (x1 * cos) + (y1 * sin)
        rot_x = int(abs(x2-x1))
        rot_y = int(abs(x1*sin))
        rot_width = int(x1 - abs(x2-x1))
        rot_height = int((y1 * cos) - (x1 * sin))
        M = cv2.getRotationMatrix2D((image_center),angle,1)
        M[0, 2] += ((x2 / 2) - image_center[0])
        M[1, 2] += ((y2 / 2) - image_center[1])
                  
        print('angle has changed')
        local_angle = angle
            
        roi_image = cv2.warpAffine(roi_image,M,(int(x2),int(y2)))  
        roi_image = roi_image[rot_y:(rot_y+rot_height) , rot_x:(rot_x+rot_width)]  #img[y:y+h, x:x+w]
        
    if chk_box2.isChecked():
        roi_image = cv2.flip(roi_image, -1)
    cv2.putText(roi_image, 'Please select a region of interest and press ENTER', (30, 30), font, 1, (2, 255, 255), 1, cv2.LINE_AA)
    cam.EndAcquisition()
    sel = cv2.selectROI(roi_image)
    cv2.destroyAllWindows()
    
    chk_box4.setCheckState(True)
    print('Cam Acuire end')
    #roi_x , y from top left corner
    roi_x, roi_y, roi_width, roi_height = sel

    ## determine Capture X parameters
    if roi_x <= 50:
        capture_x = 0
        if roi_width + roi_x <= 1340:
            capture_width = roi_width+50 + roi_x
        else: capture_width = 1440
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
            capture_height = roi_height+80+roi_y
        else: capture_height = 1080
    elif roi_y +roi_height >= 920:
        capture_y = roi_y - 80
        capture_height = (1080 - capture_y)
    elif roi_height >= 920:
        capture_y = 0
        capture_height = 1080
    else:
        capture_y = roi_y - 80
        capture_height = roi_height + 160
    
    print(sel)
    print(capture_x, capture_y, capture_width, capture_height)
    return

def save_data():
    global view_mode, no_cancel, save_mode
    global got_camera, FPS, angle
    global rot_x, rot_y, rot_width, rot_height

    local_angle = (angle+1)
    
    # [GET] camera list
    cam_list = system.GetCameras()
    print(cam_list)
    num_cameras = cam_list.GetSize()
    print('num cams',num_cameras)
    camera_start()
    if got_camera == True and num_cameras == 0:
        print('trouble')
        console_print('The camera was disconnected')
        got_camera = False
        return

    if rot_width == 0:
        console_print('Plese click on "CAMERA" and confirm the field of view')
        return
    elif roi_width == 0:
        console_print('Plese Select "ROI" and confirm number of channels')
        return        
            
    if got_camera == False:
        console_print('No Detectable Camera')
        return

    print(rot_x, rot_y, rot_width, rot_height)
    print(roi_x, roi_y, roi_width, roi_height)
    
    btn4.setEnabled(False)
    btn1.setEnabled(False)
    no_cancel = True
    FPS = int(cam_field2.text())
    print(FPS)    
    console_print('Camera View BEGIN')
    
##    if (capture_width + capture_height) > 0:
##        c_width = capture_width
##        c_height = capture_height
##    else:
    c_width = int(cam_field6.text())
    c_height = int(cam_field7.text())
    
    print(c_width,c_height)
    camera_properties(c_width , c_height, FPS)
    print('Initial FPS view = ',cam.AcquisitionFrameRate.GetValue())
    cam.BeginAcquisition()
    count = 0
    view_mode = 1
    save_mode == 0
    bar1.setMaximum(int(cam_field4.text()))
    mask = cv2.createBackgroundSubtractorMOG2(history = 3,
                                                varThreshold = 100,
                                                detectShadows = False)    
    ch = int(field3.text())
    sub_ch = []
    sum_ch1 = [0]*ch
    
    file_stem = cam_field1_1.text()
    cur_dir = Path(cwDir)
    cur_file = cur_dir/file_stem
##    suffix = '.avi'
##    out = cv2.VideoWriter(str(Path(str(cur_file)+suffix)), cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (c_width, c_height), isColor=False)


    if (local_angle != angle):
        rad = math.radians(abs(angle))
        cos = math.cos(rad)
        sin = math.sin(rad)
        
        #original frame capture dimensions
##        y1,x1 = frame.shape
        y1 = c_height
        x1 = c_width
        image_center = (x1 / 2, y1 / 2)
        
        #rotated array dimensions
        y2 = (y1 * cos) + (x1 * sin)
        x2 = (x1 * cos) + (y1 * sin)
        rot_x = int(abs(x2-x1))
        rot_y = int(abs(x1*sin))
        rot_width = int(x1 - abs(x2-x1))
        rot_height = int((y1 * cos) - (x1 * sin))

        
        #Maxtrix transformation
        M = cv2.getRotationMatrix2D((image_center),angle,1)
        M[0, 2] += ((x2 / 2) - image_center[0])
        M[1, 2] += ((y2 / 2) - image_center[1])
        
                  
        print('angle has changed')
        local_angle = angle


    ## assigning subchannels - sub_ch
    for x in range(ch+1):
        sub_ch_x = round(x*(roi_width/(ch)))
        sub_ch.append(sub_ch_x)


    if menu1.currentText() == 'AVI':
        suffix = '.avi'
        out = cv2.VideoWriter(str(Path(str(cur_file)+suffix)), 0x00000000, 25.0, (rot_width, rot_height), isColor=False)
    elif menu1.currentText() == 'MJPG':
        suffix = '.avi'
        out = cv2.VideoWriter(str(Path(str(cur_file)+suffix)), cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (rot_width, rot_height), isColor=False)
    elif menu1.currentText() == 'MP4':
        suffix = '.mp4'
        out = cv2.VideoWriter(str(Path(str(cur_file)+suffix)), 0x00000022, 25.0, (rot_width, rot_height), isColor=False)
    else:
        suffix = '.avi'
        out = cv2.VideoWriter(str(Path(str(cur_file)+suffix)), cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (rot_width, rot_height), isColor=False)
    filename = Path(str(cur_file)+suffix)
    console_print('The saved file name = '+str(filename))
    capture_times = []
    augment_times = []
    contour_times = []

    while (no_cancel and count <= int(cam_field4.text())):
    
        count +=1

        image_primary = cam.GetNextImage()
        frame = np.array(image_primary.GetNDArray())
        start = time.time() * 1000
        if chk_box2.isChecked():
            frame = cv2.flip(frame, -1)

        if angle != 0:    
            frame = cv2.warpAffine(frame,M,(int(x2),int(y2)))  
            frame_crop = frame[rot_y:(rot_y+rot_height) , rot_x:(rot_x+rot_width)]  #img[y:y+h, x:x+w]

        if angle == 0:
            frame_crop = frame
        out.write(frame_crop)
        end = time.time() * 1000
        capture_times.append(end-start)

        start = time.time() * 1000
        frame_count = frame_crop[roi_y:(roi_y + int(roi_width*110/840)) , roi_x:(roi_x + roi_width)]
        frame_mask = mask.apply(frame_count)
        frame_mask = cv2.medianBlur(frame_mask,blur_value)
        frame_mask = cv2.threshold(frame_mask, 125, 255, cv2.THRESH_BINARY)[1]
        end = time.time() * 1000
        augment_times.append(end-start)
        # find contours
        start = time.time() * 1000
        contours, hierarcy = cv2.findContours(frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # list of all the coordinates (tuples) of each cell
        coord_list = []

        # to find the coordinates of the cells
        for i in range(len(contours)):
            avg = np.mean(contours[i], axis = 0)
            coord = (int(avg[0][0]), int(avg[0][1])) ##Coord is (y,x)
##            if Show == 1:
##                cv2.circle(frame_count, coord, 10, (255, 0, 255), 1)
            ch_pos = int(math.floor((coord[0])/sub_ch[1]))
            try:
                sum_ch1[ch_pos] += 1
            except:
                #Undeclared variable?
                # error += 1
                print("error")
        end = time.time() * 1000
        contour_times.append(end-start)
        if FPS < 26 or (count%(int(FPS/5)) == 0):

            if chk_box4.isChecked() and (capture_width+capture_height) > 0:
                cv2.putText(frame_crop, 'ROI', (roi_x + 10, roi_y + 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_crop, 'Capture Size', (capture_x + 10, capture_y + 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame_crop, (roi_x,roi_y), (roi_x+roi_width, roi_y+roi_height), (255,255,255), 3,1,0)
                cv2.rectangle(frame_crop, (capture_x ,capture_y), (capture_x + capture_width, capture_y + capture_height), (255,255,255), 3,1,0)
                # draw lines on crop frame
                ## 
                for x in range(ch):
                    cv2.putText(frame_crop, str(x+1), ((roi_x+sub_ch[x]+10), (roi_y + int(roi_width*110/840) - 25)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(frame_crop, ((roi_x+sub_ch[x]),roi_y), ((roi_x+sub_ch[x]), (roi_y+int(roi_width*110/840))), (200,200,100),2)
                
            qformat = QImage.Format_Indexed8
            img = QImage(frame, int(x2), int(y2), int(x2), qformat)
            img = img.copy( rot_x, rot_y, rot_width, rot_height)    # copy(int x, int y, int width, int height)
            img = QPixmap.fromImage(img)
            
            if chk_box3.isChecked():
                img=img.scaled(round(width*(0.6))-50, round(height*(0.6))-25, aspectRatioMode = Qt.KeepAspectRatio)
            
            cv2.imshow('frame_count', frame_count)
            
            label.setPixmap(img)
            cv2.waitKey(Delay)
            
        image_primary.Release()
        bar1.setValue(count)


    out.release()
    averaged_capture = np.mean(capture_times)
    averaged_augment = np.mean(augment_times)
    averaged_contour = np.mean(contour_times)
    print("Average capture times: %f ms" % averaged_capture)
    print("Averaged augment times: %f ms" % averaged_augment)
    print("Averaged contour times: %f ms" % averaged_contour)
    cam.EndAcquisition()
    cv2.destroyAllWindows()
    print('FPS captured is =' , FPS)
    view_mode = 0
    label.setText("Hello...What's next?")
    camera_reset()
    no_cancel = False
    #set an array of sub channel dimension
    print('[RESULTS] for RUN is ', sum_ch1)
    total_sum = []
    
    total_sum.append(sum_ch1)
    
    ###write dataframes and export to an Excel file    
    check = 0
    title = []
    for j in range(len(total_sum)):
        if check < len(total_sum[j]):check = len(total_sum[j])
        title.append('Run %i '%(j+1)+str(cur_file))

    index=np.arange(0,check,1)

    for k in range(len(total_sum)):
        if len(total_sum[k]) < check:
            for l in range(len(total_sum[k]),check):
                total_sum[k].append(0)

    TTotal_sum = list(map(list, zip(*total_sum)))
    df = DataFrame(data=TTotal_sum, columns = title)
    df.to_excel(str(Path(str(cur_file)+ ".xlsx")), index=False, sheet_name="Results")

    btn1.setEnabled(True)
    btn4.setEnabled(True)     
    log_param()
    
    return

def console_print(string):
    # dd/mm/YY H:M:S
    global console_num
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    log_input = (now, string)
    console_num += 1
    console_value = (str(console_num) + ' | ' + dt_string + ' |   ' + str(string))
    console_list.insertItem(0,console_value)
    console_log.append(log_input)
    
def save():
    global filename, file_stem
    file_stem = cam_field1_1.text()
    cur_dir = Path(cwDir)
    cur_file = cur_dir/file_stem
    if menu1.currentText() == 'AVI':
        suffix = '.avi'
    elif menu1.currentText() == 'MJPG':
        suffix = '.avi'
    elif menu1.currentText() == 'MP4':
        suffix = '.mp4'
    else: suffix = '.avi'
    filename = Path(str(cur_file)+suffix)
    console_print('The saved file name = '+str(filename))    
    
def log_param():
    from datetime import date
    from datetime import datetime

    today = date.today()
    date = today.strftime("%B %d, %Y")
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    file_stem = cam_field1_1.text()
    cur_dir = Path(cwDir)
    cur_file = cur_dir/file_stem

    log=[]
    log_param_name = [cam_name1.text(), cam_name1_1.text(), cam_name1_2.text(),
                      cam_name2.text(), cam_name3.text(), cam_name4.text(),
                      cam_name5.text(), cam_name6.text(),
                      cam_name7.text(), cam_name9.text(),
                      name1.text(), name2.text(), name3.text(), 
                      name4.text(), name5.text(), name6.text()]
    log_param_value = [cam_field1.text(), cam_field1_1.text(), menu1.currentText(),
                       cam_field2.text(), cam_field3.text(), cam_field4.text(),
                       cam_field5.text(), cam_field6.text(),
                       cam_field7.text(), rotation_value(),
                       field1.text(), field2.text(), field3.text(),
                       field4.text(), field5.text(), field6.text() ]
    log.append(log_param_name)
    log.append(log_param_value)

    with open(str(Path(str(cur_file) + "_log" + ".txt")), 'w') as f:
        f.write(date + "\n")
        f.write(dt_string + "\n")
        f.write("===================================="+"\n")
        for x in range(len(log_param_name)):
            f.write(str(log_param_name[x]) + " " + str(log_param_value[x]) +"\n")
        f.write("===================================="+"\n")
            
    

##    with open('log.csv', 'wb') as csvfile:
##        filewriter = csv.writer(csvfile, delimiter=',',
##                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
##        filewriter.writerows(log_param_name)
##        filewriter.writerows(log_param_value)

    return

## Styles ##
##btn1.setStyleSheet("background-color: rgb(225,40,15);"
##                   "border-style: inset;"
##                   "min-width: 10em;"
##                   "border-radius: 4px;"
##                   "padding: 10px;")
play_btn = ("QPushButton { background-color: rgb(25,185,75); font: bold 12px;}"
            "QPushButton { border-color: rgb(40,40,15); border-width: 2px; }"
            "QPushButton { border-style: inset; border-radius: 4px; padding: 10px; }"
            "QPushButton:pressed { background-color: rgb(50,165,75) }")

cancel_btn = ("QPushButton { background-color: rgb(185,25,75); font: bold 12px;}"
            "QPushButton { border-color: rgb(40,40,15); border-width: 2px; }"
            "QPushButton { border-style: inset; border-radius: 4px; padding: 10px; }"
            "QPushButton:pressed { background-color: rgb(165,50,75) }")

save_btn = ("QPushButton { background-color: rgb(25,120,250); font: bold 12px;}"
            "QPushButton { border-color: rgb(40,40,15); border-width: 2px; }"
            "QPushButton { border-style: inset; border-radius: 4px; padding: 10px; }"
            "QPushButton:pressed { background-color: rgb(25,120,200) }")

cam_btn = ("QPushButton { background-color: rgb(200,200,200); font: bold 12px;}"
            "QPushButton { border-color: rgb(40,40,15); border-width: 2px; }"
            "QPushButton { border-style: inset; border-radius: 4px; padding: 10px; }"
            "QPushButton:pressed { background-color: rgb(180,180,180) }")


## --- Setup Camera --- ##
#Try using camera trigger instead and see if it speeds up the image capture
# [GET] system
system = PySpin.System.GetInstance()

## Always start by initializing Qt (only once per application)
app = QApplication([])
    
## Define a top-level widget to hold everything
w = QWidget()
w.setWindowTitle('The window for testing')
w.setMinimumWidth(win_length)
w.setMinimumHeight(win_width)
##w.setGeometry(0,100,win_length,win_width)  ##(x,y (downsward), lenght, height)

w2 = QWidget()



## Create some widgets to be placed inside
# ==== Tabs =====#
tab = QTabWidget()

        
# ==== Labels ==== #
name1 = QLabel('Select File(s):')
name2 = QLabel('Height of ROI:')
name3 = QLabel('Channels:')
name4 = QLabel('Blur Value (3,5 or 7):')
name5 = QLabel('Delay (ms):')
name6 = QLabel('No of frame skip:')
cam_name1 = QLabel('Select Folder:')
cam_name1_1 = QLabel('Name File:')
cam_name1_2 = QLabel('File Type:')
cam_name2 = QLabel('FPS:')
cam_name3 = QLabel('Exposure Time (us):')
cam_name4 = QLabel('No. of Frames:')
cam_name5 = QLabel('Gain:')
cam_name6 = QLabel('Camera Width:')
cam_name7 = QLabel('Camera Height:')
cam_name8 = QLabel('Show ROI')
cam_name9 = QLabel('Rotate')

label = QLabel()
label.setText('OpenCV Image')
label.setAlignment(Qt.AlignCenter)


# ==== Buttons ==== #
btn1 = QPushButton('PLAY'); btn1.setStyleSheet(play_btn); btn1.setFixedSize(80,40)
btn2 = QPushButton('CANCEL'); btn2.setStyleSheet(cancel_btn); btn2.setFixedSize(80,40)
btn3 = QPushButton('SAVE'); btn3.setStyleSheet(save_btn); btn3.setFixedSize(80,40)
btn4 = QPushButton('CAMERA'); btn4.setStyleSheet(cam_btn); btn4.setFixedSize(80,40)
btn5 = QPushButton('ROI'); btn4.setStyleSheet(cam_btn); btn4.setFixedSize(80,40)
btn6 = QPushButton('btn6'); btn4.setFixedSize(80,40)
btn7 = QPushButton('btn7'); btn4.setFixedSize(80,40)
btn8 = QPushButton('Reset Value'); btn4.setFixedSize(80,40)
cam_btn1 = QPushButton('Select Folder')
process_btn1 = QPushButton('Select File(s)')

# ==== Entries ==== #
field1 = QLineEdit('Folder / file anme')
field2 = QLineEdit(str(H))
field3 = QLineEdit(str(Channels))
field4 = QLineEdit(str(blur_value))
field5 = QLineEdit(str(Delay))
field6 = QLineEdit(str(Skip_frames))
cam_field1 =  QLineEdit(cwDir); cam_field1.setFixedSize(150,22)
cam_field1_1 =  QLineEdit(file_stem)
cam_field2 =  QLineEdit(str(FPS))
cam_field3 =  QLineEdit(str(exposure_time))
cam_field4 =  QLineEdit(str(num_frames))
cam_field5 =  QLineEdit(str(gain))
cam_field6 =  QLineEdit(str(width))
cam_field7 =  QLineEdit(str(height))

# ==== Checkbox ===== #
chk_box1 = QCheckBox('Show Video'); chk_box1.setCheckState(True); chk_box1.setTristate(False)
chk_box2 = QCheckBox('Flip Video'); chk_box2.setCheckState(False)
chk_box3 = QCheckBox('Window Scale'); chk_box3.setCheckState(True); chk_box3.setTristate(False)
chk_box4 = QCheckBox('Show ROI'); chk_box4.setCheckState(False);chk_box4.setTristate(False)
## example to detect change state checkBox.stateChanged.connect(self.enlarge_window)
## label.setScaledContents(chk_box3.isChecked())

# ==== Sliders ===== #
slide1 = QSlider(Qt.Horizontal)
slide1.setMinimum(-40)
slide1.setMaximum(40)
slide1.setValue(0)
slide1.setTickPosition(QSlider.TicksBelow) #TicksBothSides, TicksBelow
slide1.setSingleStep(1)
slide1.setTickInterval(4)
slide1.valueChanged.connect(rotation_value)

# ==== Buttons ==== #
console_list = QListWidget()
console_list.setFixedSize(win_length*0.7,100)

# ==== plot ===== #
plot = pg.PlotWidget()

# ==== Progress Bar ===== #
bar1 = QProgressBar()

# ==== Dropdown Menu ===== #
menu1 = QComboBox()
menu1.addItem('AVI')
menu1.addItem('MJPG')
menu1.addItem('MP4')

box1 = QGroupBox('Buttons')
box2 = QGroupBox('Camera Settings')
box3 = QGroupBox('Processing Parameters')
box4 = QGroupBox('View Video')
box1.setFixedSize(win_length*0.25,win_width*0.95)
##box4.setFixedSize(1510,665)

## Create a grid layout for Box 2 to manage the widgets size and position within box
box2_layout = QGridLayout()
box2_layout.addWidget(cam_btn1, 0, 0)
box2_layout.addWidget(cam_field1, 0, 1)
box2_layout.addWidget(cam_name1_1, 1, 0)
box2_layout.addWidget(cam_field1_1, 1, 1)
box2_layout.addWidget(cam_name1_2, 2, 0)
box2_layout.addWidget(menu1, 2, 1)
box2_layout.addWidget(cam_name2, 3, 0)
box2_layout.addWidget(cam_field2, 3, 1)
box2_layout.addWidget(cam_name3, 4, 0)
box2_layout.addWidget(cam_field3, 4, 1,)
box2_layout.addWidget(cam_name4, 5, 0)
box2_layout.addWidget(cam_field4, 5, 1)
box2_layout.addWidget(cam_name5, 6, 0)
box2_layout.addWidget(cam_field5, 6, 1)
box2_layout.addWidget(cam_name6, 7, 0)
box2_layout.addWidget(cam_field6, 7, 1)
box2_layout.addWidget(cam_name7, 8, 0)
box2_layout.addWidget(cam_field7, 8, 1)
box2_layout.addWidget(cam_name8, 9, 0)
box2_layout.addWidget(chk_box4, 9, 1)
box2_layout.addWidget(cam_name9, 10, 0)
box2_layout.addWidget(slide1, 10, 1)

box2.setLayout(box2_layout)


## Create a grid layout to manage the widgets size and position within box
layout = QGridLayout() ##define layout - grid type
##layout.setRowMinimumHeight(1, 0)
####layout.setColumnStretch(2, 100)
####layout.setColumnStretch(5, 100)
box3.setLayout(layout)

## Create The surrounding box widget to wrap the "layout"
windowLayout = QGridLayout() ##define layout - Box type
windowLayout.addWidget(btn2, 0, 3,1,1)
windowLayout.addWidget(btn1, 0, 1,1,1)
windowLayout.addWidget(btn3, 0, 4,1,1)
windowLayout.addWidget(btn4, 0, 0,1,1)
windowLayout.addWidget(btn5, 1, 0,1,1)
windowLayout.addWidget(btn6, 1, 1,1,1)
windowLayout.addWidget(btn7, 1, 3,1,1)
windowLayout.addWidget(btn8, 1, 4,1,1)
windowLayout.addWidget(box2,2,0,1,5)
windowLayout.addWidget(box3,4,0,1,5)
## Second parent box layout
box1.setLayout(windowLayout)

## Create The surrounding box widget to wrap the "layout"
videoLayout = QGridLayout() ##define layout - Box type
videoLayout.addWidget(label,0,0)
box4.setLayout(videoLayout)

## Main window layout
## Create The 2 surrounding box widget to wrap the "layout" of layout
bigLayout =  QGridLayout()
bigLayout.addWidget(box4,0,0,1,5)
bigLayout.addWidget(box1,0,5,0,1)
bigLayout.addWidget(bar1,1,0,1,5)  # list widget goes in bottom-left
bigLayout.addWidget(console_list,2,0,1,5)

## execute the widget layout
w.setLayout(bigLayout)

## tabs creation
w2layout = QGridLayout()
label1 = QLabel("Widget in Tab 1.")
label2 = QLabel("Widget in Tab 2.")
tab.addTab(label1, "Tab 1")
tab.addTab(label2, "Tab 2")
w2.setLayout(w2layout)
w2layout.addWidget(tab, 0, 0)

## Add widgets to the layout in their proper positions
layout.addWidget(process_btn1, 0, 0)
layout.addWidget(field1, 0, 1)

layout.addWidget(name2, 1, 0)
layout.addWidget(name3, 2, 0)
layout.addWidget(field2, 1, 1)
layout.addWidget(field3, 2, 1)

layout.addWidget(name4, 3, 0)
layout.addWidget(name5, 4, 0)
layout.addWidget(field4, 3, 1)
layout.addWidget(field5, 4, 1)

layout.addWidget(name6, 5, 0)
layout.addWidget(field6, 5, 1)
layout.addWidget(chk_box1, 6, 0)
layout.addWidget(chk_box2, 6, 1)
layout.addWidget(chk_box3, 7, 0)

## Action Mouse clicks ##
##btn1.clicked.connect(on_click_save)
camera_start()
btn1.clicked.connect(video)
btn2.clicked.connect(cancel)
btn3.clicked.connect(save_data)
btn4.clicked.connect(camera_view)
btn5.clicked.connect(roi_select)
btn7.clicked.connect(log_param)
btn8.clicked.connect(reset_parameters)
process_btn1.clicked.connect(select_files)
cam_btn1.clicked.connect(select_folder)
## menu1.activated[str].connect(self.style_choice)
## checkBox.stateChanged.connect(self.enlarge_window)

## Display the widget as a new window
w.show()
##w2.show()

## Start the Qt event loop
app.exec_()
