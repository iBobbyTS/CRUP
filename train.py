import os
from time import time
import argparse

import cv2
import torch
from torch.optim import Adam

from metrics import psnr
from model import HDRPointwiseNN
from utils import save_params, get_latest_ckpt, load_params, listdir
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
            sdr = FFmpeg2YUV(
                os.path.join(params['dataset'], file),
                bt2020_to_bt709=True, bit_depth=10,
                l_sl=params['net_input_size'], f_sl=params['net_output_size'],
                ffmpeg_path=params['ffmpeg_path']
            )
            hdr = FFmpeg2YUV(
                os.path.join(params['dataset'], file),
                bt2020_to_bt709=False, bit_depth=10,
                f_sl=params['net_output_size'],
                ffmpeg_path=params['ffmpeg_path']
            )
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
                    hdr_[j] = torch.from_numpy(hdr.read()[0]).to(device)
                sdr_lr /= 1023
                sdr_full /= 1023
                hdr_ /= 1023
                _ = (sdr_lr*255).numpy()[0]
                cv2.imwrite('/Users/ibobby/Dataset/train/sdr_lr_y.png', _[0])
                cv2.imwrite('/Users/ibobby/Dataset/train/sdr_lr_u.png', _[1])
                cv2.imwrite('/Users/ibobby/Dataset/train/sdr_lr_v.png', _[2])
                _ = (sdr_full*255).numpy()[0]
                cv2.imwrite('/Users/ibobby/Dataset/train/sdr_full_y.png', _[0])
                cv2.imwrite('/Users/ibobby/Dataset/train/sdr_full_u.png', _[1])
                cv2.imwrite('/Users/ibobby/Dataset/train/sdr_full_v.png', _[2])
                _ = (hdr_*255).numpy()[0]
                cv2.imwrite('/Users/ibobby/Dataset/train/hdr_full_y.png', _[0])
                cv2.imwrite('/Users/ibobby/Dataset/train/hdr_full_u.png', _[1])
                cv2.imwrite('/Users/ibobby/Dataset/train/hdr_full_v.png', _[2])
                exit(1)
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
        ckpt_model_filename = f'ckpt_{epoch}.pth'
        ckpt_model_path = os.path.join(params['ckpt_path'], ckpt_model_filename)
        state = save_params(model.state_dict(), {**params, 'Time': epoch_time})
        torch.save(state, ckpt_model_path)
        model.to(device).train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-low', type=int, dest='net_input_size')
    parser.add_argument('-full', type=int, dest='net_output_size')
    parser.add_argument('-lr', type=float)
    parser.add_argument('-bs', type=int, dest='batch_size')
    parser.add_argument('-ff', type=str, dest='ffmpeg_path')

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
        'dataset_suffix': '',
        'ffmpeg_path': ''
    }
    params.update(parser.parse_args().__dict__)

    train(params=params)
