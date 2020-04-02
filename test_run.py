###############################################################################
#                                                                             #
#                                                                             #
#  (c) Simon Wenkel                                                           #
#  released under a 3-Clause BSD license                                      #
#                                                                             #
#                                                                             #
#                                                                             #
###############################################################################


import argparse
import os
###############################################################################
# import libraries                                                            #
#                                                                             #
import time
from queue import Queue
from threading import Thread

import cv2
from tqdm import tqdm


#                                                                             #
###############################################################################


###############################################################################
# classes and functions                                                       #
#                                                                             #
def parseARGS() -> dict:
    """
    Argument parser
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("-vf", "--videofile",
    #                     required=True,
    #                     type=str)
    parser.add_argument("-uCPU", "--useCPU",
                        default="True",
                        type=str)
    parser.add_argument("-uCU", "--useCUDA",
                        default="True",
                        type=str)
    args = parser.parse_args()
    config = {}
    config["videofile"] = "Raw Video Output 10x Inv-L.avi"
    if args.useCPU == "True":
        config["useCPU"] = True
    else:
        config["useCPU"] = False
    if args.useCUDA == "True":
        config["useCUDA"] = True
    else:
        config["useCUDA"] = False
    return config


class video_queue:
    """
    Video input queue

    based on this implementation:
    https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
    by Adrian Rosebrock released under MIT license
    """

    def __init__(self,
                 file: str,
                 target: str = "cpu",
                 queue_size: int = 500):
        """


        """
        self.stream = cv2.VideoCapture(file)
        if target == "cpu":
            self.target = target
        elif target == "cuda":
            self.target = target
        else:
            raise Exception("target must be of type 'cpu' or 'cuda'!")
        self.stopped = False
        self.Queue = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

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

                if self.target == "cuda":
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
            time.sleep(0.1)
            tries += 1

        return self.Queue.qsize() > 0

    def stop(self):
        """

        """
        self.stopped = True
        self.thread.join()


def apply_BGS_CPU(algorithm,
                  video_file: str):
    """
    Apply CPU implementation of an algorithm
    """
    video_stream = video_queue(video_file, target="cpu").start()
    start_time = time.time()
    frame_counter = 0
    while video_stream.more():
        frame = video_stream.read()
        frame = algorithm.apply(frame)
        frame_counter += 1
    fps = frame_counter / (time.time() - start_time)
    return fps


def apply_BGS_GPU(algorithm,
                  video_file: str):
    """
    Apply CUDA implementation of an algorithm
    """
    cuda_stream = cv2.cuda_Stream()
    video_stream = video_queue(video_file, target="cuda").start()
    start_time = time.time()
    frame_counter = 0
    while video_stream.more():
        frame = video_stream.read()
        frame2 = algorithm.apply(frame, 0.1, cuda_stream)
        frame_counter += 1
    fps = frame_counter / (time.time() - start_time)
    return fps


#                                                                             #
###############################################################################


###############################################################################
# CONSTANTS                                                                   #
#                                                                             #
cpu_algorithms = {}
cpu_algorithms["MOG2"] = cv2.createBackgroundSubtractorMOG2(history=120,
                                                            varThreshold=250,
                                                            detectShadows=True)
cpu_algorithms["KNN"] = cv2.createBackgroundSubtractorKNN(history=120,
                                                          dist2Threshold=400,
                                                          detectShadows=True)

gpu_algorithms = {}
gpu_algorithms["MOG"] = cv2.cuda.createBackgroundSubtractorMOG(history=120)
gpu_algorithms["MOG2"] = cv2.cuda.createBackgroundSubtractorMOG2(history=120,
                                                                 varThreshold=250,
                                                                 detectShadows=True)


#                                                                             #
###############################################################################

def main():
    print("=" * 80)
    print("Benchmarking OpenCV Background Subtractors")
    print("-" * 10)
    print("Parsing arguments")
    config = parseARGS()
    print("-" * 10)
    print("Create folder to store results")
    os.makedirs("../results/", exist_ok=True)
    results_file = open("../results/results_python.csv", "w")
    video_file = config["videofile"]
    print("-" * 10)
    print("Starting Benchmark")
    print("-" * 10)
    for i in tqdm(range(20)):
        if config["useCPU"]:
            for bgs in cpu_algorithms:
                fps = apply_BGS_CPU(cpu_algorithms[bgs],
                                    video_file)
                results = video_file + ", " + bgs + ", " + bgs + " (CPU, Python)" + ", CPU, Python, " + str(fps) + "\n"
                results_file.write(results)
                results_file.flush()
                os.fsync(results_file)

        if config["useCUDA"]:
            for bgs in gpu_algorithms:
                fps = apply_BGS_GPU(gpu_algorithms[bgs],
                                    video_file)
                results = video_file + ", " + bgs + ", " + bgs + " (CUDA, Python)" + ", CUDA, Python, " + str(
                    fps) + "\n"
                results_file.write(results)
                results_file.flush()
                os.fsync(results_file)
    print("=" * 80)


if __name__ == "__main__":
    main()