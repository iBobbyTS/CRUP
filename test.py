import subprocess
import numpy as np
import skimage.exposure
import torch

from model import HDRPointwiseNN
from utils import load_params
from dataset import FFmpeg2YUV


def test(ckpt, args: dict):
    # State_dict
    state_dict = torch.load(ckpt)['weight']
    state_dict, params = load_params(state_dict)
    # Params
    params.update(args)
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Video
    cap = FFmpeg2YUV(
        params['test_image'],
        bt2020_to_bt709=False, bit_depth=10,
        return_ori=True, l_sl=256,
        ffmpeg_path=''
    )
    pipe = subprocess.Popen([
        'ffmpeg',
        '-f', 'rawvideo', '-pix_fmt', 'yuv444p16le'
        '-s', '%dx%d' % (cap.y_height, cap.y_width), '-r', '25',
        '-i', '-',
        '-c:v', 'hevc', '-tag:v', 'hvc1', '-pix_fmt', 'yuv420p10le',
        args['test_out']
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Model
    torch.set_grad_enabled(False)
    model = HDRPointwiseNN(params=params)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    for i in range(cap.num_frames):
        low, full = cap.read()
        low = torch.from_numpy(low).unsqueeze(0)/1023
        img = model(low, full)
        img = (img.cpu().detach().numpy()).transpose(0, 2, 3, 1)[0]
        img = skimage.exposure.rescale_intensity(img, out_range=(0.0, 65535.0)).astype(np.uint16)
        pipe.stdin.write(img.tobytes())
    pipe.terminate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--checkpoint', type=str, help='model state path')
    parser.add_argument('--input', type=str, dest='test_image', help='image path')
    parser.add_argument('--output', type=str, dest='test_out', help='output image path')

    args = vars(parser.parse_args())

    test(args['checkpoint'], args)
