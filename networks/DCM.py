import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DCM(nn.Module):
    def __init__(self, psf_kernel_size=71, num_psfs=3, psi_range=(-4., 10.)):
        super(DCM, self).__init__()
        self.psf_kernel_size = psf_kernel_size
        self.num_psfs = num_psfs
        self.precomputed_params = {}
        self.psi_range = psi_range
        self.act = nn.ReLU()

    def load_pre_compute_data(self, precompute_data_path):
        from scipy.io import loadmat
        mat_file = loadmat(precompute_data_path)['data'][0][0]
        precomputed_params = {
            'radius': np.asscalar(mat_file[0]),
            'ref_wl': np.asscalar(mat_file[1]),
            'focus_point': np.asscalar(mat_file[2]),
            'Psi': np.asarray(mat_file[3])[0],
            'min_dist': np.asscalar(mat_file[4]),
            'psf_kernels': mat_file[5].reshape(
                self.num_psfs, self.psf_kernel_size, self.psf_kernel_size, -1),
            'map_interval': np.asscalar(mat_file[6]),
            'num_psi_classes': np.asscalar(mat_file[7]),
            'downsample_factor': np.asscalar(mat_file[8]),
            'dn_fs': np.asscalar(mat_file[9]),
        }
        precomputed_params.update({'psi_resolution': np.round(
            float(precomputed_params['Psi'][-1] - precomputed_params['Psi'][-2]), decimals=1)})
        self.precomputed_params = precomputed_params

        self.register_buffer('psf_kernels', torch.from_numpy(precomputed_params['psf_kernels']))
        self.register_buffer('psi_values', torch.from_numpy(precomputed_params['Psi']).float())
        self.register_buffer('int_psi_values',
                             torch.Tensor(
                                 [psi_val for psi_val in self.psi_values if psi_val.int().float() == psi_val]))

    def psf_conv(self, sub_img, psf_kernels, conv_bias=None):
        """
        Notes:
             - unsqueeze(0) is done since F.conv2d expects [B, C, H, W] so adding Batch dim.
             - Casting the input to double to match the psf kernels weights type
        """
        pad_val = tuple([self.psf_kernel_size // 2 for _ in range(4)])
        pad_input = F.pad(sub_img, pad_val, mode='replicate')
        blur_sub_img = F.conv2d(pad_input, psf_kernels.unsqueeze(1), bias=conv_bias, groups=self.num_psfs)
        return blur_sub_img.squeeze(0)

    def _extract_psf_kernels(self, depth):
        psi_ind = torch.isclose(depth, self.psi_values,
                                atol=self.precomputed_params['psi_resolution'] / 2).nonzero(as_tuple=True)[0][0].item()
        return self.psf_kernels[:, :, :, psi_ind]

    def _extract_psf_kernels2(self, psi_ind):
        return self.psf_kernels[:, :, :, psi_ind]

    def scale_value_to_psi_range(self, unscaled_psi_map):
        depth_min, depth_max = self.psi_range
        scaled_psi_map = unscaled_psi_map * (depth_max - depth_min) + depth_min
        return scaled_psi_map

    def _quantize_psi_values(self, scaled_psi_map):
        H = scaled_psi_map.shape[-2]
        W = scaled_psi_map.shape[-1]
        indicator_map = torch.isclose(scaled_psi_map.squeeze(0).repeat(len(self.psi_values), 1, 1),
                                      self.psi_values.view(-1, 1, 1).repeat(1, H, W),
                                      atol=self.precomputed_params['psi_resolution'] / 2)  # [Psi_res, H, W]
        quantized_scaled_psi_map = torch.Tensor(1, H, W).to(scaled_psi_map.device)
        for h in range(H):
            for w in range(W):
                quantized_scaled_psi_map[:, h, w] = \
                    self.psi_values[indicator_map[:, h, w].nonzero(as_tuple=True)[0][0].int()]
        return quantized_scaled_psi_map

    def forward(self, x, epoch, debug=False):
        rgb = x[:, :3, :, :]
        scaled_weights = self.scale_value_to_psi_range(x[:, 3:, :, :])
        int_planes = self.int_psi_values.view(1, -1, 1, 1).expand(
            (1, len(self.int_psi_values), *scaled_weights.shape[-2:]))
        B, _, H, W = rgb.shape
        blur_imgs = torch.zeros(B, len(self.int_psi_values), self.num_psfs, H, W, device=x.device)
        for psi_ind, psi in enumerate(self.int_psi_values):
            relevant_psf_kernels = self._extract_psf_kernels(psi)
            blur_sub_img = self.psf_conv(rgb, relevant_psf_kernels)
            blur_imgs[:, psi_ind, :, :, :] = blur_sub_img

        interpolation_weights = self.act(1 - torch.abs(scaled_weights - int_planes))
        out = torch.einsum('nijkl, njmkl -> nmkl', interpolation_weights.unsqueeze(1), blur_imgs)

        if debug and epoch % 100 == 0:
            self.log_depth_weights(interpolation_weights.detach().cpu())

        return out
