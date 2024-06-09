from __future__ import print_function

from networks.skip import skip
from utils.common_utils import *
from utils.SSIM import SSIM
from networks.DCM import DCM
from utils.psf_utils import depth_read, calc_average_error_for_depth

import argparse
import os
import glob
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from kornia.losses import TotalVariation, InverseDepthSmoothnessLoss
from torchvision.io import read_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity
import numpy as np
import torch
import random
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=2501, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 512], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[71, 71], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="data", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results", help='path to results')
parser.add_argument('--save_frequency', type=int, default=500, help='frequency to save results')
parser.add_argument('--index', type=int, default=0, help='image index in dataset')
parser.add_argument('--input_index', type=str, default='noise', help='input codes, noise for DIP, fourier for PIP')

parser.add_argument('--do_reg', action='store_true', default=False, help='Do perturbation noise or not')
parser.add_argument('--alpha_depth', type=float, default=0, help='weighting for depth reguralizations')
parser.add_argument('--depth_reg_type', type=str, default='TV', choices=['TV', 'ID'])

opt = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)

warnings.filterwarnings("ignore")

input_source = glob.glob(os.path.join(opt.data_path, 'rgb', '*.png'))
input_source.sort()

sharp_source = glob.glob(os.path.join(opt.data_path, 'inputs', '*.*'))
sharp_source.sort()

psi_maps_source = glob.glob(os.path.join(opt.data_path, 'GT', '*.dpt'))
psi_maps_source.sort()

save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

index = opt.index

for f in input_source[index:index + 1]:
    INPUT = opt.input_index
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('fish') != -1:
        opt.kernel_size = [41, 41]
    if imgname.find('flower') != -1:
        opt.kernel_size = [25, 25]
    if imgname.find('house') != -1:
        opt.kernel_size = [51, 51]
    if imgname.find('maskImg') != -1:
        opt.kernel_size = [71, 71]

    # Imaging system constants
    PSI_INT_VALS = 15
    DEPTH_MIN = -4.0
    DEPTH_MAX = 10.0

    img = read_image(path_to_image).type(dtype)
    img /= 255.0
    img_size = img.shape

    if sharp_source:
        sharp_img = read_image(sharp_source[index]).float()
        sharp_img = torchvision.transforms.ToTensor()(Image.open(sharp_source[index]).convert('RGB'))
        sharp_img_np = sharp_img.permute(1, 2, 0).numpy()

    if psi_maps_source:
        psi_map_ref = depth_read(psi_maps_source[index])
        psi_map_ref = np.expand_dims(psi_map_ref, -1)
        psi_map_ref = (psi_map_ref - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

    print(imgname)

    padw, padh = opt.kernel_size[0] - 1, opt.kernel_size[1] - 1
    opt.img_size[0], opt.img_size[1] = img_size[1] + padh, img_size[2] + padw

    img.unsqueeze_(0)

    freq_dict_img = {
        'method': 'log',
        'cosine_only': False,
        'n_freqs': 8,
        'base': 2 ** (8 / (8 - 1)),
        }
    input_depth = freq_dict_img['n_freqs'] * 4

    net_input = get_input(input_depth, INPUT, (img_size[1], img_size[2]), freq_dict=freq_dict_img).type(dtype)

    # Generator
    ksize = 3 if INPUT == 'noise' else 1
    net = skip(input_depth, 4,
               num_channels_down=[128, 128, 128, 128, 128],
               num_channels_up=[128, 128, 128, 128, 128],
               num_channels_skip=[16, 16, 16, 16, 16],
               upsample_mode='bilinear',
               filter_skip_size=1, filter_size_down=ksize, filter_size_up=ksize,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params: %d' % s)
    print(net)

    # DCM
    fwd_model = DCM()
    fwd_model.load_pre_compute_data(os.path.join(opt.data_path, 'data.mat'))
    fwd_model.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    if opt.depth_reg_type == 'TV':
        depth_reg_func = TotalVariation()
    elif opt.depth_reg_type == 'ID':
        depth_reg_func = InverseDepthSmoothnessLoss()
    else:
        raise ValueError('Depth regularization type {} not supported'.format(opt.depth_reg_type))

    depth_reg = depth_reg_func.type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[4000, 5000], gamma=0.5)  # learning rates

    net_input_saved = net_input.detach().clone()
    exp_weight = 0.99
    depth_alpha = opt.alpha_depth

    for step in tqdm(range(num_iter)):
        if opt.do_reg:
            net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(
                net_input_saved.data).normal_()
        else:
            net_input = net_input_saved

        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        implicit_model_outputs = net(net_input)
        fwd_model_inputs = implicit_model_outputs

        out_rgb = fwd_model(fwd_model_inputs, step)

        rgb_size = out_rgb.shape
        cropw = rgb_size[2] - img_size[1]
        croph = rgb_size[3] - img_size[2]
        out_rgb = out_rgb[:, :, cropw // 2:cropw // 2 + img_size[1], croph // 2:croph // 2 + img_size[2]]

        if step < 500:
            if opt.depth_reg_type == 'TV':
                reg = depth_reg(implicit_model_outputs[:, 3:, :, :])
            else:  # ID regularization
                reg = depth_reg(implicit_model_outputs[:, 3:, :, :], img)

            total_loss = mse(out_rgb, img) + depth_alpha * reg

        else:
            total_loss = (1 - ssim(out_rgb, img))

        total_loss.backward()
        optimizer.step()

        if step % opt.save_frequency == 0:
            B, C, H, W = out_rgb.shape
            out_x = fwd_model_inputs[:, :3, :, :]
            out_psi = fwd_model_inputs[:, 3:, :, :]
            out_x_np = out_x[0].permute(1, 2, 0).detach().cpu().numpy()
            out_rgb_np = out_rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()
            psi_np = out_psi[0].permute(1, 2, 0).cpu().detach().numpy()
            psi_np_norm = psi_np * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN

            blur_psnr = np.array([psnr(img_np[i], out_rgb_np[i]) for i in range(B)])
            sharp_psnr = psnr(sharp_img_np, out_x_np)
            sharp_ssim = structural_similarity(sharp_img_np, out_x_np, multichannel=True)
            depth_error_in_meters = calc_average_error_for_depth(psi_np, psi_map_ref)

            print(f'Depth error: {depth_error_in_meters} [m]')
            print(f'All-In-Focus PSNR: {sharp_psnr} [dB]')
            print(f'All-In-Focus SSIM: {sharp_ssim}')

            fig = plt.figure(figsize=(12, 6))
            im = plt.imshow(psi_np_norm)
            plt.colorbar(im, fraction=0.046, pad=0.04)

            fig_for_paper = plt.figure()
            vmax = np.percentile(psi_np, 95)
            plt.imshow(psi_np, cmap='magma', vmax=vmax)
            plt.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            os.makedirs(opt.save_path, exist_ok=True)
            plt.savefig(os.path.join(opt.save_path, f'{imgname}_depth_plot_iter_{step}.png'), bbox_inches='tight',
                        pad_inches=0)
            plt.imsave(os.path.join(opt.save_path, f'{imgname}_sharp_iter_{step}.png'), out_x_np)
            plt.imsave(os.path.join(opt.save_path, f'{imgname}_depth_raw_{step}.png'), psi_np[:, :, 0])
            plt.close('all')
