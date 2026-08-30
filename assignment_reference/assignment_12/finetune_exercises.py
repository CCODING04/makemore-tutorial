import math
def lora_params(layer_dims, r): return sum(r*(o+i) for o,i in layer_dims)
def lora_ratio(layer_dims, r, base_params): return lora_params(layer_dims, r)/base_params
def merged_weight(W,A,B,alpha,r): return W + (alpha/r)*(B@A)
def merge_changes_output(W,A,B,alpha,r,x,tol=1e-6):
    import torch
    y0 = W@x + (alpha/r)*(B@(A@x)); W2 = merged_weight(W,A,B,alpha,r)
    return torch.allclose(y0, W2@x, atol=tol)
def initial_delta_norm(A,B): return float((B@A).pow(2).sum().sqrt())
def qlora_vram_gb(base_params_billion, quant_bits=4, lora_params=20_000_000):
    return (base_params_billion*1e9*quant_bits/8 + lora_params*12)/1e9
