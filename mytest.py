import subprocess
import numpy
import cv2
pipe = subprocess.Popen(
    '/Users/ibobby/root/bin/ffmpeg -v error '
    '-ss 15 -i /Users/ibobby/Dataset/hdr/jTdI5igs1C0.mkv '
    '-frames 1 '
    '-vf zscale=t=linear:npl=100,format=yuv420p10le,'
    'zscale=p=bt709,tonemap=tonemap=clip:desat=0,'
    'zscale=t=bt709:m=bt709:r=tv '
    '-f rawvideo -pix_fmt yuv444p '
    '-'.split(' '),
    stdout=subprocess.PIPE, bufsize=1000000000
)

y, u, v = numpy.frombuffer(pipe.stdout.read(1920*1080*3), dtype=numpy.uint8).reshape(3, 1080, 1920)

cv2.imwrite('/Users/ibobby/Dataset/y_8.png', y)
cv2.imwrite('/Users/ibobby/Dataset/u_8.png', u)
cv2.imwrite('/Users/ibobby/Dataset/v_8.png', v)

pipe.terminate()
