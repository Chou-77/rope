import torch
import torch.nn as nn
from absl import logging
import numpy as np
import math
from tqdm import tqdm


def get_sde(name, **kwargs):
    # --- ADDED FOR ZTSNR: 允許在 config 中透過名字加上 _ztsnr 來啟用 ---
    is_ztsnr = name.endswith('_ztsnr')
    base_name = name.replace('_ztsnr', '')

    if base_name == 'vpsde':
        sde = VPSDE(**kwargs)
    elif base_name == 'vpsde_cosine':
        sde = VPSDECosine(**kwargs)
    else:
        raise NotImplementedError

    if is_ztsnr:
        return ZTSNR_SDE(sde)
    return sde


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


class SDE(object):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """

    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, t):
        raise NotImplementedError

    def cum_beta(self, t):  # the variance of xt|x0
        raise NotImplementedError

    def cum_alpha(self, t):
        raise NotImplementedError

    def snr(self, t):  # signal noise ratio
        raise NotImplementedError

    def nsr(self, t):  # noise signal ratio
        raise NotImplementedError

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def sample(self, x0, t_init=0):  # sample from q(xn|x0), where n is uniform
        t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + stp(std, eps)
        return t, eps, xt


# --- ADDED FOR ZTSNR: 處理連續 SDE 的強制歸零 ---
class ZTSNR_SDE(SDE):
    def __init__(self, base_sde: SDE):
        self.base_sde = base_sde
        # 計算 t=1 也就是最後一步時的原本 alpha 值，準備用來做平移縮放
        t_max = torch.tensor([1.0])
        self.sqrt_alpha_T = base_sde.cum_alpha(t_max).sqrt().item()

    def cum_alpha(self, t):
        sqrt_alpha = self.base_sde.cum_alpha(t).sqrt()
        # Lin et al. 的連續時間重標定公式
        rescaled_sqrt_alpha = (sqrt_alpha - self.sqrt_alpha_T) / (1.0 - self.sqrt_alpha_T)
        # 確保不會因為浮點數誤差變成負數，限制最小值
        rescaled_sqrt_alpha = torch.clamp(rescaled_sqrt_alpha, min=1e-5)
        return rescaled_sqrt_alpha ** 2

    def cum_beta(self, t):
        return 1.0 - self.cum_alpha(t)

    def snr(self, t):
        alpha = self.cum_alpha(t)
        return alpha / (1.0 - alpha)

    def nsr(self, t):
        alpha = self.cum_alpha(t)
        return (1.0 - alpha) / alpha

    def drift(self, x, t):
        return self.base_sde.drift(x, t)

    def diffusion(self, t):
        return self.base_sde.diffusion(t)


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        # 0 <= t <= 1
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def drift(self, x, t):
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):  # beta(t)
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def squared_diffusion_integral(self, s, t):  # \int_s^t beta(tau) d tau
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

    def skip_beta(self, s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
        return 1. - self.skip_alpha(s, t)

    def skip_alpha(self, s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()

    def cum_beta(self, t):
        return self.skip_beta(0, t)

    def cum_alpha(self, t):
        return self.skip_alpha(0, t)

    def nsr(self, t):
        return self.squared_diffusion_integral(0, t).expm1()

    def snr(self, t):
        return 1. / self.nsr(t)

    def __str__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'

    def __repr__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'


class VPSDECosine(SDE):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """

    def __init__(self, s=0.008):
        self.s = s
        self.F = lambda t: torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.F0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    def drift(self, x, t):
        ft = - torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi / 2
        return stp(ft, x)

    def diffusion(self, t):
        return (torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi) ** 0.5

    def cum_beta(self, t):  # the variance of xt|x0
        return 1 - self.cum_alpha(t)

    def cum_alpha(self, t):
        return self.F(t) / self.F0

    def snr(self, t):  # signal noise ratio
        Ft = self.F(t)
        return Ft / (self.F0 - Ft)

    def nsr(self, t):  # noise signal ratio
        Ft = self.F(t)
        return self.F0 / Ft - 1

    def __str__(self):
        return 'vpsde_cosine'

    def __repr__(self):
        return 'vpsde_cosine'


class ScoreModel(object):
    r"""
        The forward process is q(x_[0,T])
    """

    def __init__(self, nnet: nn.Module, pred: str, sde: SDE, T=1):
        assert T == 1
        self.nnet = nnet
        self.pred = pred
        self.sde = sde
        self.T = T
        print(f'ScoreModel with pred={pred}, sde={sde}, T={T}')

    def predict(self, xt, conditions, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        return self.nnet(xt, conditions, t * 999)  # follow SDE

    def noise_pred(self, xt, conditions, t):
        pred = self.predict(xt, conditions, t)
        if self.pred == 'noise_pred':
            noise_pred = pred
        elif self.pred == 'x0_pred':
            noise_pred = - self.sde.snr(t).sqrt().unsqueeze(1).unsqueeze(1).unsqueeze(1) * pred + self.sde.cum_beta(
                t).rsqrt().unsqueeze(1).unsqueeze(1).unsqueeze(1) * xt
        # --- ADDED FOR V-PRED ---
        elif self.pred == 'v_pred':
            alpha = self.sde.cum_alpha(t)
            beta = self.sde.cum_beta(t)
            noise_pred = stp(alpha ** 0.5, pred) + stp(beta ** 0.5, xt)
        else:
            raise NotImplementedError
        return noise_pred

    def x0_pred(self, xt, conditions, t):
        pred = self.predict(xt, conditions, t)
        if self.pred == 'noise_pred':
            x0_pred = self.sde.cum_alpha(t).rsqrt() * xt - self.sde.nsr(t).sqrt() * pred
        elif self.pred == 'x0_pred':
            x0_pred = pred
        # --- ADDED FOR V-PRED ---
        elif self.pred == 'v_pred':
            alpha = self.sde.cum_alpha(t)
            beta = self.sde.cum_beta(t)
            x0_pred = stp(alpha ** 0.5, xt) - stp(beta ** 0.5, pred)
        else:
            raise NotImplementedError
        return x0_pred

    def score(self, xt, conditions, t):
        cum_beta = self.sde.cum_beta(t)
        noise_pred = self.noise_pred(xt, conditions, t)
        return stp(-cum_beta.rsqrt(), noise_pred)


class ReverseSDE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) dw
    """

    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, conditions, t):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, conditions, t)
        return drift - stp(diffusion ** 2, score)

    def diffusion(self, t):
        return self.sde.diffusion(t)


class ODE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt
    """

    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, conditions, t):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, conditions, t)
        return drift - 0.5 * stp(diffusion ** 2, score)

    def diffusion(self, t):
        return 0


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


@torch.no_grad()
def euler_maruyama(rsde, x_init, sample_steps, conditions, eps=1e-3, T=1, trace=None, verbose=False):
    r"""
    The Euler Maruyama sampler for reverse SDE / ODE
    See `Score-Based Generative Modeling through Stochastic Differential Equations`
    """
    assert isinstance(rsde, ReverseSDE) or isinstance(rsde, ODE)
    print(f"euler_maruyama with sample_steps={sample_steps}")
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x_init)
    x = x_init
    if trace is not None:
        trace.append(x)
    for s, t in tqdm(list(zip(timesteps, timesteps[1:]))[::-1], disable=not verbose, desc='euler_maruyama'):
        drift = rsde.drift(x, conditions, t)
        diffusion = rsde.diffusion(t)
        dt = s - t
        mean = x + drift * dt
        sigma = diffusion * (-dt).sqrt()
        x = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
        if trace is not None:
            trace.append(x)
        statistics = dict(s=s, t=t, sigma=sigma.item())
        logging.debug(dct2str(statistics))
    return x


# --- MODIFIED FOR V-PRED ---
# def LSimple(score_model: ScoreModel, x0, conditions, pred='noise_pred'):
#     t, noise, xt = score_model.sde.sample(x0)
#
#     # 統一交給神經網路預測
#     model_out = score_model.predict(xt, conditions, t)
#
#     if pred == 'noise_pred':
#         return mos(noise - model_out)
#     elif pred == 'x0_pred':
#         return mos(x0 - model_out)
#     elif pred == 'v_pred':
#         # Target v = sqrt(alpha)*noise - sqrt(beta)*x0
#         alpha = score_model.sde.cum_alpha(t)
#         beta = score_model.sde.cum_beta(t)
#         target_v = stp(alpha ** 0.5, noise) - stp(beta ** 0.5, x0)
#         return mos(target_v - model_out)
#     else:
#         raise NotImplementedError(pred)

def LSimple(score_model: ScoreModel, x0, conditions, pred='v_pred', gamma=5.0):
    t, noise, xt = score_model.sde.sample(x0)
    model_out = score_model.predict(xt, conditions, t)
    alpha = score_model.sde.cum_alpha(t)
    beta = score_model.sde.cum_beta(t)
    snr = alpha / (beta + 1e-5)
    if pred == 'noise_pred':
        target = noise
        weight = torch.clamp(snr, max=gamma)
    elif pred == 'x0_pred':
        target = x0
        weight = torch.clamp(snr, max=gamma) / snr
    elif pred == 'v_pred':
        target = stp(alpha ** 0.5, noise) - stp(beta ** 0.5, x0)
        weight = torch.clamp(snr, max=gamma) / (snr + 1.0)
    else:
        raise NotImplementedError(pred)
    squared_error = (target - model_out) ** 2
    loss_per_sample = squared_error.mean(dim=[1, 2, 3])
    weight = weight.view(-1)
    final_loss = (loss_per_sample * weight).mean()
    return final_loss