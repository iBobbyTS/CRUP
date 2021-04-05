import os
from time import time
import argparse

import cv2
import torch
from torch.optim import Adam

from metrics import psnr
from model_y_uv import HDRPointwiseNN
# from mytest import HDRPointwiseNN
from utils import get_latest_ckpt, load_params, listdir
from dataset import FFmpeg2YUV


def train(params=None):
    os.makedirs(params['ckpt_path'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_model = HDRPointwiseNN(n_channel=1, net_input_size=params['net_input_size'])
    uv_model = HDRPointwiseNN(n_channel=2, net_input_size=params['net_input_size'])
    ckpt = get_latest_ckpt(params['ckpt_path'])
    if ckpt:
        print('Loading previous state:', ckpt)
        state_dict = torch.load(ckpt)
        state_dict, params = map(state_dict.get, ('weight', 'model_params'))
        y_model.load_state_dict(state_dict['y'])
        uv_model.load_state_dict(state_dict['uv'])
    y_model.to(device)
    uv_model.to(device)

    y_mseloss = torch.nn.MSELoss()
    uv_mseloss = torch.nn.MSELoss()
    y_optimizer = Adam(y_model.parameters(), params['lr'])
    uv_optimizer = Adam(uv_model.parameters(), params['lr'])

    files = listdir(params['dataset'])
    total_batch_count = 0
    for f in files:
        cap = cv2.VideoCapture(f"{params['dataset']}/{f}")
        total_batch_count += int(cap.get(7) // params['batch_size'])
        cap.release()

    for epoch in range(1, params['epochs'] + 1):
        epoch_time = 0
        current_batch_count = 1
        y_model.train()
        uv_model.train()
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
                y_optimizer.zero_grad()
                uv_optimizer.zero_grad()

                # Train
                y_res = y_model(sdr_lr[:, [0]], sdr_full)
                uv_res = uv_model(sdr_lr[:, 1:], sdr_full)

                # MSE loss
                y_loss = y_mseloss(y_res, hdr_[:, [0]])
                y_loss.backward()
                y_loss = y_loss.item()
                uv_loss = uv_mseloss(uv_res, hdr_[:, 1:])
                uv_loss.backward()
                uv_loss = uv_loss.item()
                # PSNR Loss
                y_psnr = psnr(y_res, hdr_[:, [0]]).item()
                uv_psnr = psnr(uv_res, hdr_[:, 1:]).item()

                y_optimizer.step()
                uv_optimizer.step()
                time_spent = time() - batch_start_time
                epoch_time += time_spent
                print(
                    "\r"
                    f"Epoch {epoch}/{params['epochs']} | "
                    f"Batch {current_batch_count}/{total_batch_count} | "
                    "MSE Loss(Y/UV): %.02e/%.02e | "
                    "PSNR Loss(Y/UV): %.02f/%.02f | "
                    "Estimated Time Lest: %.02f | "
                    "Epoch Time: %.02f | "
                    "Batch Time: %.02f"
                    "" % (
                        y_loss, uv_loss, y_psnr, uv_psnr,
                        epoch_time/current_batch_count*(total_batch_count-current_batch_count),
                        epoch_time, time_spent
                    ),
                    end='', flush=True
                )
                current_batch_count += 1
        torch.save(
            {
                'weight': {
                    'y': y_model.state_dict(),
                    'uv': uv_model.state_dict()
                },
                'model_params': {
                    'Time': epoch_time,
                    **params
                }
            },
            os.path.join(params['ckpt_path'], f'ckpt_{epoch}.pth')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-low', type=int, dest='net_input_size', default=256)
    parser.add_argument('-full', type=int, dest='net_output_size', default=512)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-ep', type=int, dest='epochs', default=10)
    parser.add_argument('-bs', type=int, dest='batch_size', default=4)
    parser.add_argument('-ff', type=str, dest='ffmpeg_path')
    parser.add_argument('-md', type=str, dest='ckpt_model_path')

    params = {
        'ckpt_path': './model_weights',
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
        'ffmpeg_path': '',
        'ckpt_model_path': ''
    }
    params.update(parser.parse_args().__dict__)

    train(params=params)
