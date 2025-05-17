# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import math
import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')


#----------------------------------------------------------------------------
# DPM-Solver-v3 from the paper
# "DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics",
# using a training-free version.

class NoiseScheduleEDM:
    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.zeros_like(t).to(torch.float64)

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.ones_like(t).to(torch.float64)

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t.to(torch.float64)

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        return -torch.log(t).to(torch.float64)

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return torch.exp(-lamb).to(torch.float64)


def weighted_cumsumexp_trapezoid(a, x, b, cumsum=True):
    # ∫ b*e^a dx
    # Input: a,x,b: shape (N+1,...)
    # Output: y: shape (N+1,...)
    # y_0 = 0
    # y_n = sum_{i=1}^{n} 0.5*(x_{i}-x_{i-1})*(b_{i}*e^{a_{i}}+b_{i-1}*e^{a_{i-1}}) (n from 1 to N)

    assert x.shape[0] == a.shape[0] and x.ndim == a.ndim
    if b is not None:
        assert a.shape[0] == b.shape[0] and a.ndim == b.ndim

    a_max = np.amax(a, axis=0, keepdims=True)

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    out = 0.5 * (x[1:] - x[:-1]) * (tmp[1:] + tmp[:-1])
    if not cumsum:
        return np.sum(out, axis=0) * np.exp(a_max)
    out = np.cumsum(out, axis=0)
    out *= np.exp(a_max)
    return np.concatenate([np.zeros_like(out[[0]]), out], axis=0)


def weighted_cumsumexp_trapezoid_torch(a, x, b, cumsum=True):

    assert x.shape[0] == a.shape[0] and x.ndim == a.ndim
    if b is not None:
        assert a.shape[0] == b.shape[0] and a.ndim == b.ndim

    a_max = torch.amax(a, dim=0, keepdims=True)

    if b is not None:
        tmp = b * torch.exp(a - a_max)
    else:
        tmp = torch.exp(a - a_max)

    out = 0.5 * (x[1:] - x[:-1]) * (tmp[1:] + tmp[:-1])
    if not cumsum:
        return torch.sum(out, dim=0) * torch.exp(a_max)
    out = torch.cumsum(out, dim=0)
    out *= torch.exp(a_max)
    return torch.concat([torch.zeros_like(out[[0]]), out], dim=0)


def index_list(lst, index):
    new_lst = []
    for i in index:
        new_lst.append(lst[i])
    return new_lst


class DPM_Solver_v3:
    def __init__(self, statistics_steps, noise_schedule, steps=10, t_start=None, t_end=None, skip_type="logSNR", device="cuda"):
        # precompute
        self.device = device
        self.noise_schedule = noise_schedule
        self.steps = steps
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        self.statistics_steps = statistics_steps
        ts = noise_schedule.marginal_lambda(self.get_time_steps("logSNR", t_T, t_0, self.statistics_steps, "cpu")).numpy()[:, None, None, None]
        self.ts = torch.from_numpy(ts).cuda()
        self.lambda_T = self.ts[0].cpu().item()
        self.lambda_0 = self.ts[-1].cpu().item()
        shape = (statistics_steps + 1, 1, 1, 1)
        l = np.ones(shape)
        s = np.zeros(shape)
        b = np.zeros(shape)
        z = np.zeros_like(l)
        o = np.ones_like(l)
        L = weighted_cumsumexp_trapezoid(z, ts, l)
        S = weighted_cumsumexp_trapezoid(z, ts, s)

        I = weighted_cumsumexp_trapezoid(L + S, ts, o)
        B = weighted_cumsumexp_trapezoid(-S, ts, b)
        C = weighted_cumsumexp_trapezoid(L + S, ts, B)
        self.l = torch.from_numpy(l).cuda()
        self.s = torch.from_numpy(s).cuda()
        self.b = torch.from_numpy(b).cuda()
        self.L = torch.from_numpy(L).cuda()
        self.S = torch.from_numpy(S).cuda()
        self.I = torch.from_numpy(I).cuda()
        self.B = torch.from_numpy(B).cuda()
        self.C = torch.from_numpy(C).cuda()

        # precompute timesteps
        if skip_type == "logSNR" or skip_type == "time_uniform" or skip_type == "time_quadratic":
            self.timesteps = self.get_time_steps(skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            self.indexes = self.convert_to_indexes(self.timesteps)
            self.timesteps = self.convert_to_timesteps(self.indexes, device)
        elif skip_type == "edm":
            self.indexes, self.timesteps = self.get_timesteps_edm(N=steps, device=device)
            self.timesteps = self.convert_to_timesteps(self.indexes, device)
        else:
            raise ValueError(f"Unsupported timestep strategy {skip_type}")

        # store high-order exponential coefficients (lazy)
        self.exp_coeffs = {}

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def convert_to_indexes(self, timesteps):
        logSNR_steps = self.noise_schedule.marginal_lambda(timesteps)
        indexes = list(
            (self.statistics_steps * (logSNR_steps - self.lambda_T) / (self.lambda_0 - self.lambda_T)).round().cpu().numpy().astype(np.int64)
        )
        return indexes

    def convert_to_timesteps(self, indexes, device):
        logSNR_steps = self.lambda_T + (self.lambda_0 - self.lambda_T) * torch.Tensor(indexes).to(device) / self.statistics_steps
        return self.noise_schedule.inverse_lambda(logSNR_steps)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_timesteps_edm(self, N, device):
        """Constructs the noise schedule of Karras et al. (2022)."""

        rho = 7.0

        sigma_min: float = np.exp(-self.lambda_0)
        sigma_max: float = np.exp(-self.lambda_T)
        ramp = np.linspace(0, 1, N + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        lambdas = torch.Tensor(-np.log(sigmas)).to(device)
        timesteps = self.noise_schedule.inverse_lambda(lambdas)

        indexes = list((self.statistics_steps * (lambdas - self.lambda_T) / (self.lambda_0 - self.lambda_T)).round().cpu().numpy().astype(np.int64))
        return indexes, timesteps

    def get_g(self, f_t, i_s, i_t):
        return torch.exp(self.S[i_s] - self.S[i_t]) * f_t - torch.exp(self.S[i_s]) * (self.B[i_t] - self.B[i_s])

    def compute_exponential_coefficients_high_order(self, i_s, i_t, order=2):
        key = (i_s, i_t, order)
        if key in self.exp_coeffs.keys():
            coeffs = self.exp_coeffs[key]
        else:
            n = order - 1
            a = self.L[i_s : i_t + 1] + self.S[i_s : i_t + 1] - self.L[i_s] - self.S[i_s]
            x = self.ts[i_s : i_t + 1]
            b = (self.ts[i_s : i_t + 1] - self.ts[i_s]) ** n / math.factorial(n)
            coeffs = weighted_cumsumexp_trapezoid_torch(a, x, b, cumsum=False)
            self.exp_coeffs[key] = coeffs
        return coeffs

    def compute_high_order_derivatives(self, n, lambda_0n, g_0n, pseudo=False):
        # return g^(1), ..., g^(n)
        if pseudo:
            D = [[] for _ in range(n + 1)]
            D[0] = g_0n
            for i in range(1, n + 1):
                for j in range(n - i + 1):
                    D[i].append((D[i - 1][j] - D[i - 1][j + 1]) / (lambda_0n[j] - lambda_0n[i + j]))

            return [D[i][0] * math.factorial(i) for i in range(1, n + 1)]
        else:
            R = []
            for i in range(1, n + 1):
                R.append(torch.pow(lambda_0n[1:] - lambda_0n[0], i))
            R = torch.stack(R).t()
            B = (torch.stack(g_0n[1:]) - g_0n[0]).reshape(n, -1)
            shape = g_0n[0].shape
            solution = torch.linalg.inv(R) @ B
            solution = solution.reshape([n] + list(shape))
            return [solution[i - 1] * math.factorial(i) for i in range(1, n + 1)]

    def multistep_predictor_update(self, x_lst, eps_lst, time_lst, index_lst, t, i_t, order=1, pseudo=False):
        # x_lst: [..., x_s]
        # eps_lst: [..., eps_s]
        # time_lst: [..., time_s]
        ns = self.noise_schedule
        n = order - 1
        indexes = [-i - 1 for i in range(n + 1)]
        x_0n = index_list(x_lst, indexes)
        eps_0n = index_list(eps_lst, indexes)
        time_0n = torch.FloatTensor(index_list(time_lst, indexes)).cuda()
        index_0n = index_list(index_lst, indexes)
        lambda_0n = ns.marginal_lambda(time_0n)
        alpha_0n = ns.marginal_alpha(time_0n)
        sigma_0n = ns.marginal_std(time_0n)

        alpha_s, alpha_t = alpha_0n[0], ns.marginal_alpha(t)
        i_s = index_0n[0]
        x_s = x_0n[0]
        g_0n = []
        for i in range(n + 1):
            f_i = (sigma_0n[i] * eps_0n[i] - self.l[index_0n[i]] * x_0n[i]) / alpha_0n[i]
            g_i = self.get_g(f_i, index_0n[0], index_0n[i])
            g_0n.append(g_i)
        g_0 = g_0n[0]
        x_t = (
            alpha_t / alpha_s * torch.exp(self.L[i_s] - self.L[i_t]) * x_s
            - alpha_t * torch.exp(-self.L[i_t] - self.S[i_s]) * (self.I[i_t] - self.I[i_s]) * g_0
            - alpha_t * torch.exp(-self.L[i_t]) * (self.C[i_t] - self.C[i_s] - self.B[i_s] * (self.I[i_t] - self.I[i_s]))
        )
        if order > 1:
            g_d = self.compute_high_order_derivatives(n, lambda_0n, g_0n, pseudo=pseudo)
            for i in range(order - 1):
                x_t = (
                    x_t
                    - alpha_t
                    * torch.exp(self.L[i_s] - self.L[i_t])
                    * self.compute_exponential_coefficients_high_order(i_s, i_t, order=i + 2)
                    * g_d[i]
                )
        return x_t

    def multistep_corrector_update(self, x_lst, eps_lst, time_lst, index_lst, order=1, pseudo=False):
        # x_lst: [..., x_s, x_t]
        # eps_lst: [..., eps_s, eps_t]
        # lambda_lst: [..., lambda_s, lambda_t]
        ns = self.noise_schedule
        n = order - 1
        indexes = [-i - 1 for i in range(n + 1)]
        indexes[0] = -2
        indexes[1] = -1
        x_0n = index_list(x_lst, indexes)
        eps_0n = index_list(eps_lst, indexes)
        time_0n = torch.FloatTensor(index_list(time_lst, indexes)).cuda()
        index_0n = index_list(index_lst, indexes)
        lambda_0n = ns.marginal_lambda(time_0n)
        alpha_0n = ns.marginal_alpha(time_0n)
        sigma_0n = ns.marginal_std(time_0n)

        alpha_s, alpha_t = alpha_0n[0], alpha_0n[1]
        i_s, i_t = index_0n[0], index_0n[1]
        x_s = x_0n[0]
        g_0n = []
        for i in range(n + 1):
            f_i = (sigma_0n[i] * eps_0n[i] - self.l[index_0n[i]] * x_0n[i]) / alpha_0n[i]
            g_i = self.get_g(f_i, index_0n[0], index_0n[i])
            g_0n.append(g_i)
        g_0 = g_0n[0]
        x_t_new = (
            alpha_t / alpha_s * torch.exp(self.L[i_s] - self.L[i_t]) * x_s
            - alpha_t * torch.exp(-self.L[i_t] - self.S[i_s]) * (self.I[i_t] - self.I[i_s]) * g_0
            - alpha_t * torch.exp(-self.L[i_t]) * (self.C[i_t] - self.C[i_s] - self.B[i_s] * (self.I[i_t] - self.I[i_s]))
        )
        if order > 1:
            g_d = self.compute_high_order_derivatives(n, lambda_0n, g_0n, pseudo=pseudo)
            for i in range(order - 1):
                x_t_new = (
                    x_t_new
                    - alpha_t
                    * torch.exp(self.L[i_s] - self.L[i_t])
                    * self.compute_exponential_coefficients_high_order(i_s, i_t, order=i + 2)
                    * g_d[i]
                )
        return x_t_new

    @torch.no_grad()
    def sample(self, model_fn, x, order, p_pseudo, use_corrector, c_pseudo, lower_order_final, return_intermediate=False):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        steps = self.steps
        cached_x = []
        cached_model_output = []
        cached_time = []
        cached_index = []
        indexes, timesteps = self.indexes, self.timesteps
        step_p_order = 0

        for step in range(1, steps + 1):
            cached_x.append(x)
            cached_model_output.append(self.noise_prediction_fn(x, timesteps[step - 1]))
            cached_time.append(timesteps[step - 1])
            cached_index.append(indexes[step - 1])
            if use_corrector:
                step_c_order = step_p_order + c_pseudo
                if step_c_order > 1:
                    x_new = self.multistep_corrector_update(
                        cached_x, cached_model_output, cached_time, cached_index, order=step_c_order, pseudo=c_pseudo
                    )
                    sigma_t = self.noise_schedule.marginal_std(cached_time[-1])
                    l_t = self.l[cached_index[-1]]
                    N_old = sigma_t * cached_model_output[-1] - l_t * cached_x[-1]
                    cached_x[-1] = x_new
                    cached_model_output[-1] = (N_old + l_t * cached_x[-1]) / sigma_t
            if step < order:
                step_p_order = step
            else:
                step_p_order = order
            if lower_order_final:
                step_p_order = min(step_p_order, steps + 1 - step)
            t = timesteps[step]
            i_t = indexes[step]

            x = self.multistep_predictor_update(cached_x, cached_model_output, cached_time, cached_index, t, i_t, order=step_p_order, pseudo=p_pseudo)

        if return_intermediate:
            return x, cached_x
        else:
            return x


def model_wrapper(denoier, noise_schedule):
    def noise_pred_fn(x, t, **kwargs):
        output = denoier(x, t, **kwargs)
        alpha_t, sigma_t = noise_schedule.marginal_alpha(t), noise_schedule.marginal_std(t)
        return (x - alpha_t[:, None, None, None] * output) / sigma_t[:, None, None, None]

    def model_fn(x, t):
        return noise_pred_fn(x, t).to(torch.float64)

    return model_fn


def get_dpmv3_sampler(num_steps=32, sigma_min=0.002, sigma_max=80):
    ns = NoiseScheduleEDM()

    dpm_solver_v3 = DPM_Solver_v3(250, ns, steps=num_steps, t_start=sigma_max, t_end=sigma_min, skip_type="edm", device="cuda")

    def dpm_solver_v3_sampler(model_fn, z):
        with torch.no_grad():
            x = dpm_solver_v3.sample(model_fn, z, order=2, p_pseudo=False, use_corrector=True, c_pseudo=False, lower_order_final=True)
        return x

    return dpm_solver_v3_sampler


def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, sampler=None,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)
    
    if sampler is not None:
        ns = NoiseScheduleEDM()
        noise_pred_fn = model_wrapper(denoise, ns)
        return sampler(noise_pred_fn, noise * sigma_max)

    #----------------------------------------------------------------------------
    # EDM sampler from the paper
    # "Elucidating the Design Space of Diffusion-Based Generative Models",
    # extended to support guidance.

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    dpmv3               = False,                # Use DPM-Solver-v3 for faster inference?
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        net.use_fp16 = True
        net.force_fp32 = False
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if net.label_dim > 0:
        all_class_labels = torch.arange(net.label_dim, device=device, dtype=torch.int64).repeat(len(seeds) // net.label_dim).tensor_split(num_batches)
        rank_class_labels = all_class_labels[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    sampler = get_dpmv3_sampler(sampler_kwargs["num_steps"], sampler_kwargs["sigma_min"], sampler_kwargs["sigma_max"]) if dpmv3 else None
    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rank_class_labels[batch_idx]]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=edm_sampler, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, sampler=sampler, **sampler_kwargs)
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--net',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--gnet',                     help='Guiding network pickle filename', metavar='PATH|URL',             type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)
@click.option('--dpmv3',                    help='Use DPM-Solver-v3 for faster inference',                          is_flag=True)

def cmdline(**opts):
    """Generate random images using the given model.
    """
    opts = dnnlib.EasyDict(opts)

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
