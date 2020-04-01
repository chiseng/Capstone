import cv2
import imutils
import numpy as np
from imutils.video import FileVideoStream

"""
TESTS: 

Normal loading, fast loading.
Normal bgsubtract, cuda bgsubtract, numba bgsubtract

"""
filepath = "Raw Video Output 10x Inv-L.avi"

def cv_bgsub_slow(filepath):
    video_stream = cv2.VideoCapture(filepath)
    while True:
        (grabbed, frame) = video_stream.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # resize the frame and convert it to grayscale (while still
        # retaining 3 channels)
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])
        # display a piece of text to the frame (so we can benchmark
        # fairly against the fast method)
        # cv2.putText(
        #     frame,
        #     "Slow Method",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (0, 255, 0),
        #     2,
        # )
        # show the frame and update the FPS counter
    #  cv2.imshow("Frame", frame)
    # cv2.waitKey(0)
    return frame


def fast_bgsub_thread(filepath):
    video_stream = FileVideoStream(filepath).start()
    while video_stream.read() is not None:
        frame = video_stream.read()
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])

    return frame

# start = time.time()
# frame_1 = cv_bgsub_slow(filepath)
# print(f"{np.round(time.time() - start,3)}s")
#
# start = time.time()
# frame_2 = fast_bgsub_thread(filepath)
# print( f"{np.round(time.time() - start,3)}s")





