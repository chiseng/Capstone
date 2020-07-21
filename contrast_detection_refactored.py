import time

import PySpin
import cv2
import numpy as np


# get system00
def capture_video(exposure, gain, blur_value, fps, duration, avi=True):
    system = PySpin.System.GetInstance()

    # get camera list
    cam_list : PySpin.CameraList = system.GetCameras()
    # use primary camera
    cam: PySpin.CameraPtr = cam_list.GetByIndex(0)

    # initialize camera
    cam.Init()

    ### Camera setting ###
    # Set acquisition mode to continuous
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    # turn off auto exposure
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

    # Ensure desired exposure time does not exceed the maximum
    exposure_time_to_set = exposure  # in us
    exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
    cam.ExposureTime.SetValue(exposure_time_to_set)

    # set width to maximum
    cam.OffsetX.SetValue(0)
    ##cam.Width.SetValue(cam.Width.GetMax())

    # set height to value
    ##cam.Height.SetValue(cam.Height.GetMax())
    cam.OffsetY.SetValue(0)
    ##cam.Height.SetValue(cam.Height.GetMax())

    # set gain
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(gain)

    # frame rate
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(fps)
    framerate = cam.AcquisitionFrameRate.GetValue()
    # cap = cv2.VideoCapture(1)
    # ret,frame = cap.read()
    # sel = cv2.selectROI(frame)

    cam.BeginAcquisition()
    roi_image = cam.GetNextImage()
    roi_image = np.array(roi_image.GetNDArray())
    sel = cv2.selectROI(roi_image)
    roi_x, roi_y, roi_width, roi_height = sel
    cv2.destroyAllWindows()
    print(
        "ROI vairables:\n x %i\n y %i\n width %i\n height %i\n"
        % (roi_x, roi_y, roi_width, roi_height)
    )
    cam.EndAcquisition()
    return


def main():
    Capture_FPS = 25  # Capture FPS used for show and test
    duration = 20
    exposure = 20000  # time in us
    gain = 0  # gain value 0-40
    font = cv2.FONT_HERSHEY_SIMPLEX
    blur_value = 7
    capture_video(exposure, gain, blur_value, Capture_FPS, duration)


if __name__ == "__main__":
    main()
