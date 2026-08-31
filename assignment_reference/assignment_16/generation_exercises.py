import math
import torch
import torch.nn.functional as F

def q_sample(x0, alphas_cumprod, t, noise):
    s = alphas_cumprod[t].view(-1, 1).sqrt()
    return s * x0 + (1 - alphas_cumprod[t].view(-1, 1)).sqrt() * noise

def signal_ratio(t, betas):
    alphas = 1 - betas
    return math.sqrt(torch.cumprod(alphas, dim=0)[t].item())

def cfg(eps_uncond, eps_cond, w):
    return eps_uncond + w * (eps_cond - eps_uncond)

def img2img_start_step(strength, num_inference_steps):
    return math.floor(num_inference_steps * strength)

def decoupled_cross_attn(Q, K_txt, V_txt, K_ref, V_ref, scale):
    d = Q.shape[-1]
    attn = lambda X, K, V: F.softmax(X @ K.transpose(-2, -1) / d ** 0.5, -1) @ V
    return attn(Q, K_txt, V_txt) + scale * attn(Q, K_ref, V_ref)
