import subprocess
import numpy
import cv2
pipe = subprocess.Popen(
    'ffmpeg -v error -i /Users/ibobby/Dataset/resolution_test/1080p.mov -f rawvideo -pix_fmt yuv444p -'.split(' '),
    stdout=subprocess.PIPE, bufsize=1000000000
)

# y = numpy.frombuffer(pipe.stdout.read(1920*1080), dtype=numpy.uint8).reshape(1080, 1920)
# u = numpy.frombuffer(pipe.stdout.read(1920*1080), dtype=numpy.uint8).reshape(1080, 1920)
# v = numpy.frombuffer(pipe.stdout.read(1920*1080), dtype=numpy.uint8).reshape(1080, 1920)
yuv = numpy.frombuffer(pipe.stdout.read(1920*1080*3), dtype=numpy.uint8).reshape(3, 1080, 1920)
# y = yuv[:1080]
# u, v = yuv[1080:].reshape(2, 540, 960)
y, u, v = yuv
# print(f)
cv2.imwrite('/Users/ibobby/Dataset/y.png', y)
cv2.imwrite('/Users/ibobby/Dataset/u.png', u)
cv2.imwrite('/Users/ibobby/Dataset/v.png', v)

pipe.terminate()
