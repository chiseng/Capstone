import time
from queue import Queue
from threading import Thread

import cv2


class video_queue:
    """
    Video input queue

    based on this implementation:
    https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
    by Adrian Rosebrock released under MIT license
    """

    def __init__(self,
                 file: str,
                 y1: int,
                 H: int,
                 x1: int,
                 x2: int,
                 queue_size: int = 500):
        """


        """
        self.stream = cv2.VideoCapture(file)
        self.stopped = False
        self.Queue = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.y1 = y1
        self.x1 = x1
        self.x2 = x2
        self.H = H

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
                # frame = frame[self.y1:(self.y1 + self.H), self.x1:self.x2]
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
            time.sleep(0.01)
            tries += 1

        return self.Queue.qsize() > 0

    def stop(self):
        """

        """
        self.stopped = True
        self.thread.join()