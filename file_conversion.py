import pathlib
import subprocess as sp
import time

import cv2
import imageio
import numpy as np
from imutils.video import FileVideoStream


class Timer:
    def __init__(self, label, decimals=1, announce_start=False):
        self.announce_start = announce_start
        self.label = label
        self.decimals = decimals

    def get_printout(self, message: str) -> str:
        return str({self.__class__.__name__: (self.label, message)})

    def __enter__(self):
        self.start = time.time()
        if self.announce_start:
            print(self.get_printout("start"))

    def __exit__(self, type, value, traceback):
        duration = time.time() - self.start
        duration = round(duration, self.decimals)
        print(self.get_printout(f"{duration} s"))


def read_video(path_video: str) -> np.ndarray:
    assert pathlib.Path(path_video).exists()
    print("Reading video")
    buf = []
    video_stream = FileVideoStream(path_video).start()
    count = 0
    while video_stream.more():
        count += 1
        frame = video_stream.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf.append(frame)
        print(count)
    return buf



def stacked_frames(frames: np.ndarray) -> np.ndarray:
    print("stacking")
    model = cv2.createBackgroundSubtractorMOG2(history=3, varThreshold=100, detectShadows=False)
    # return np.stack([model.apply(f) for f in frames])
    return np.stack([f for f in frames])

def write_video_avi(frames, fps):
    out_file = "test_vid.avi"
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'PIM1'), fps,
                          (frames[0].shape[1], frames[0].shape[0]), False)
    for f in frames:
        out.write(f)
    out.release()

def imio(frames, fps):
    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer = imageio.get_writer("test2.avi", fps=fps)
    for f in frames:
        writer.append_data(f)

def ffmpeg_writer(frames, count, fps):
    ffmpeg_bin = r'C:\ffmpeg-20200729-cbb6ba2-win64-static\bin\ffmpeg.exe'
    #ffmpeg_bin = '/usr/bin/ffmpeg'
    command = [ffmpeg_bin,
               '-y',  # (optional) overwrite output file if it exists
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{frames[0].shape[1]}x{frames[0].shape[0]}',  # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', str(fps),  # frames per second
               '-i', '-',  # The imput comes from a pipe
               '-an',  # Tells FFMPEG not to expect any audio
               '-vcodec', 'mpeg4',
               'test_ff.avi']
    proc = sp.Popen(command, stdin=sp.PIPE)
    env = lmdb.open('test1_lmdb', readonly=True)
    with env.begin() as txn:
        for idx in range(count + 1):
            proc.stdin.write(txn.get(str(idx).encode("ascii")).tostring())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0: raise sp.CalledProcessError(proc.returncode, command)


def main(path_video_in=r"C:\Users\Me\Desktop\capstone\WBC286 InvL-Pillars -350mbar 150fps 29-11-2019 v3.4.avi", out_dir="output_frames"):
    print("Converting...")
    v = cv2.VideoCapture(path_video_in)
    v.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    fps = int(v.get(cv2.CAP_PROP_FPS))
    video = read_video(path_video_in)
    output = stacked_frames(video)
    print("Frame conversion after background subtraction for demonstration")
    with Timer("ffmpeg conversion",decimals=7):
        ffmpeg_writer(output)
    with Timer("avi conversion", decimals=7):
        write_video_avi(output, fps)
    with Timer("imageio", decimals=7):
        imio(output, fps)
    with Timer("NPY file save", decimals=7):
        np.save("video_frames", output)
    with Timer("NPY compressed", decimals=7):
        np.savez("video_frames",output)
        # np.savez_compressed("video_frames", output)


if __name__ == "__main__":
    main()
