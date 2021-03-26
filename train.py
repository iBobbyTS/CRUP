import os
from time import time

import cv2
import torch
from torch.optim import Adam

from metrics import psnr
from model import HDRPointwiseNN
from utils import save_params, get_latest_ckpt, load_params, listdir
from test import test
from dataset import FFmpeg2YUV


def train(params=None):
    os.makedirs(params['ckpt_path'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HDRPointwiseNN(params=params)
    ckpt = get_latest_ckpt(params['ckpt_path'])
    if ckpt:
        print('Loading previous state:', ckpt)
        state_dict = torch.load(ckpt)
        state_dict, _ = load_params(state_dict)
        model.load_state_dict(state_dict)
    model.to(device)

    mseloss = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), params['lr'])

    files = listdir(params['dataset'])
    total_batch_count = 0
    for f in files:
        cap = cv2.VideoCapture(f"{params['dataset']}/{f}")
        total_batch_count += int(cap.get(7) // params['batch_size'])
        cap.release()

    for epoch in range(params['epochs']):
        epoch_time = 0
        current_batch_count = 0
        model.train()
        for file in files:
            sdr = FFmpeg2YUV(f"{params['dataset']}/{file}", hdr=False, bit_depth=10, ffmpeg_path='/Users/ibobby/root/bin')
            hdr = FFmpeg2YUV(f"{params['dataset']}/{file}", hdr=True, bit_depth=10, ffmpeg_path='/Users/ibobby/root/bin')
            for i in range(sdr.num_frames // params['batch_size']):
                batch_start_time = time()
                sdr_lr = torch.empty(
                    params['batch_size'], 3, params['net_input_size'], params['net_input_size'],
                    dtype=torch.float32, device=device
                )
                sdr_full = torch.empty(
                    params['batch_size'], 3, params['net_output_size'], params['net_output_size'],
                    dtype=torch.float32, device=device
                )
                hdr_ = torch.empty_like(sdr_full)
                for j in range(params['batch_size']):
                    sdr_lr[j], sdr_full[j] = [torch.from_numpy(_).to(device) for _ in sdr.read()]
                    hdr_[j] = torch.from_numpy(hdr.read()).to(device)
                sdr_lr /= 1023
                sdr_full /= 1023
                hdr_ /= 1023
                optimizer.zero_grad()

                res = model(sdr_lr, sdr_full)

                loss = mseloss(res, hdr_)
                loss.backward()
                loss = loss.item()
                _psnr = psnr(res, hdr_).item()

                optimizer.step()
                time_spent = time() - batch_start_time
                print(
                    "\r"
                    f"Epoch {epoch}/{params['epochs']} | "
                    f"Batch {current_batch_count}/{total_batch_count} | "
                    "MSE Loss: %.02e | "
                    "PSNR Loss: %.02f | "
                    "Epoch Time: %.02f | "
                    "Batch Time: %.02f"
                    "" % (
                        loss, _psnr, epoch_time, time_spent
                    ),
                    end='', flush=True
                )
                epoch_time += time_spent
                current_batch_count += 1
        model.eval().cpu()
        ckpt_model_filename = "ckpt_" + str(epoch) + ".pth"
        ckpt_model_path = os.path.join(params['ckpt_path'], ckpt_model_filename)
        state = save_params(model.state_dict(), params)
        torch.save(state, ckpt_model_path)
        test(ckpt_model_path)
        model.to(device).train()


if __name__ == '__main__':
    """
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--ckpt-path', type=str, default='./ch', help='Model checkpoint path')
    parser.add_argument('--test-image', type=str, dest="test_image", help='Test image path')
    parser.add_argument('--test-out', type=str, default='out.png', dest="test_out", help='Output test image path')

    parser.add_argument('--luma-bins', type=int, default=8)
    parser.add_argument('--channel-multiplier', default=1, type=int)
    parser.add_argument('--spatial-bin', type=int, default=16)
    parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--ckpt-interval', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='', help='Dataset path with input/output dirs', required=True)
    parser.add_argument('--dataset-suffix', type=str, default='',
                        help='Add suffix to input/output dirs. Useful when train on different dataset image sizes')

    params = vars(parser.parse_args())
    """
    params = {
        'ckpt_path': './model_weights',
        'test_image': 'test_image',
        'test_out': 'out.png',
        'luma_bins': 8,
        'channel_multiplier': 1,
        'spatial_bin': 16,
        'batch_norm': False,
        'net_input_size': 256,
        'net_output_size': 512,
        'lr': 1e-4,
        'batch_size': 4,
        'epochs': 10,
        'log_interval': 10,
        'ckpt_interval': 100,
        'dataset': '/Users/ibobby/Dataset/hdr',
        'dataset_suffix': ''
    }

    train(params=params)
