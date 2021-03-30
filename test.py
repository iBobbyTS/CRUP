import subprocess
import torch
import cv2
from skimage import exposure
import numpy

from model import HDRPointwiseNN
from utils import load_params
from dataset import FFmpeg2YUV


def test(ckpt, args: dict):
    # State_dict
    state_dict = torch.load(ckpt)
    state_dict, params = load_params(state_dict)
    # Params
    params.update(args)
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Video
    cap = FFmpeg2YUV(
        params['test_image'],
        bt2020_to_bt709=False, bit_depth=8,
        return_ori=True, l_sl=256,
        ffmpeg_path=''
    )
    pipe = subprocess.Popen([
        'ffmpeg',
        '-f', 'rawvideo', '-pix_fmt', 'yuv444p10le',
        '-s', '%dx%d' % (cap.y_width, cap.y_height), '-r', '1',
        '-i', '-',
        '-c:v', 'hevc', '-tag:v', 'hvc1', '-pix_fmt', 'yuv420p10le',
        '-color_primaries', '9', '-color_trc', '16', '-colorspace', '9',
        args['test_out'], '-y'
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Model
    torch.set_grad_enabled(False)
    model = HDRPointwiseNN(params=params)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    for i in range(cap.num_frames):
        low, full = cap.read()
        low = torch.from_numpy(low).unsqueeze(0)/255
        full = torch.from_numpy(full).unsqueeze(0)/255
        # cv2.imwrite('/Users/ibobby/Dataset/resolution_test/1080p_low_y.png', (full.squeeze(0) * 255)[0].numpy())
        # cv2.imwrite('/Users/ibobby/Dataset/resolution_test/1080p_low_u.png', (full.squeeze(0) * 255)[1].numpy())
        # cv2.imwrite('/Users/ibobby/Dataset/resolution_test/1080p_low_v.png', (full.squeeze(0) * 255)[2].numpy())
        img = model(low, full)
        img = img[0].cpu().detach().numpy()
        img = exposure.rescale_intensity(img, out_range=(0.0, 1023.0)).astype(numpy.uint16)
        # print(img.shape, img.dtype)
        # cv2.imwrite('/Users/ibobby/Dataset/resolution_test/1080p_outY.png', (img / 4)[0])
        # cv2.imwrite('/Users/ibobby/Dataset/resolution_test/1080p_outU.png', (img / 4)[1])
        # cv2.imwrite('/Users/ibobby/Dataset/resolution_test/1080p_outV.png', (img / 4)[2])
        pipe.stdin.write(img.tobytes())
    pipe.terminate()
    cap.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--checkpoint', type=str, help='model state path')
    parser.add_argument('--input', type=str, dest='test_image', help='image path')
    parser.add_argument('--output', type=str, dest='test_out', help='output image path')

    args = vars(parser.parse_args())

    test(args['checkpoint'], args)
