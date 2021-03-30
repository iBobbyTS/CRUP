import subprocess
import os

import cv2
import numpy


class FFmpeg2YUV:
    def __init__(
        self,
        path,
        bt2020_to_bt709=False, bit_depth=8,
        return_ori=False,
        l_sl=None, f_sl=None,
        ffmpeg_path=''
    ):
        self.bit_depth = bit_depth
        self.return_lr = False if l_sl is None else True
        self.return_full = False if f_sl is None else True
        self.return_ori = return_ori
        cap = cv2.VideoCapture(path)
        self.y_height, self.y_width, self.num_frames = map(lambda x: int(cap.get(x)), (
            cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_COUNT
        ))
        cap.release()
        self.uv_height = self.y_height // 2
        self.uv_width = self.y_width // 2
        self.read_amount = 1 if bit_depth == 8 else 2
        self.y_read_amount = self.y_width * self.y_height * self.read_amount
        self.uv_read_amount = self.uv_width * self.uv_height * self.read_amount * 2
        self.l_sl = l_sl
        self.f_sl = f_sl
        self.dtype = numpy.uint8 if bit_depth == 8 else numpy.uint16
        cmd = [
            os.path.join(ffmpeg_path, 'ffmpeg'),
            '-loglevel', 'error',
            '-i', path,
            *(['-vf',
               'zscale=t=linear:npl=100,format=yuv420p10le,'
               'zscale=p=bt709,tonemap=tonemap=clip:desat=0,'
               'zscale=t=bt709:m=bt709:r=tv'] if bt2020_to_bt709 else []),
            '-f', 'rawvideo',
            '-pix_fmt', f"yuv420p{'' if bit_depth == 8 else f'{bit_depth}le'}",
            '-'
        ]
        self.pipe = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, bufsize=int(self.y_width * self.y_height * self.read_amount)
        )

    def read(self):
        # read_Y = numpy.frombuffer(
        #     self.pipe.stdout.read(self.y_read_amount), dtype=self.dtype
        # ).reshape(self.y_height, self.y_width)
        # read_UV = numpy.frombuffer(
        #     self.pipe.stdout.read(self.y_read_amount // 2), dtype=self.dtype
        # ).reshape(self.uv_height, self.uv_width, 2)
        read = numpy.frombuffer(
            self.pipe.stdout.read(self.y_height * self.y_width * 3 // 2), dtype=self.dtype
        ).reshape(self.y_height * 3 // 2, self.y_width)
        read_Y = read[:self.y_height]
        read_UV = read[self.y_height:].reshape(2, self.uv_height, self.uv_width).transpose((1, 2, 0))
        returning = []
        if self.return_lr:
            lr_Y = cv2.resize(read_Y, (self.l_sl, self.l_sl), interpolation=cv2.INTER_CUBIC)
            lr_UV = cv2.resize(read_UV, (self.l_sl, self.l_sl), interpolation=cv2.INTER_CUBIC)
            lr_UV = numpy.transpose(lr_UV, (2, 0, 1))
            lr = numpy.stack((lr_Y, *lr_UV)).astype(numpy.float32)
            returning.append(lr)
        if self.return_full:
            full_Y = cv2.resize(read_Y, (self.f_sl, self.f_sl), interpolation=cv2.INTER_CUBIC)
            full_UV = cv2.resize(read_UV, (self.f_sl, self.f_sl), interpolation=cv2.INTER_CUBIC)
            full_UV = numpy.transpose(full_UV, (2, 0, 1))
            full = numpy.stack((full_Y, *full_UV)).astype(numpy.float32)
            returning.append(full)
        if self.return_ori:
            yuv444_UV = cv2.resize(read_UV, read_Y.shape[::-1], interpolation=cv2.INTER_CUBIC)
            yuv444_UV = numpy.transpose(yuv444_UV, (2, 0, 1))
            yuv444 = numpy.stack((read_Y, *yuv444_UV)).astype(numpy.float32)
            returning.append(yuv444)
        return returning

    def close(self):
        self.pipe.terminate()
