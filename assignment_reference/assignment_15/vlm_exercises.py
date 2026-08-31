import math
def patch_tokens(h, w, patch): return math.ceil(h/patch) * math.ceil(w/patch)
def vit_out_shape(n_tokens, embed_dim): return (n_tokens, embed_dim)
def infonce_loss(f_img, f_txt, scale):
    logits = scale * f_img @ f_txt.T
    labels = torch.arange(len(f_img), device=logits.device)
    return 0.5*(F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
def mlp2x_params(vision_dim, llm_dim):
    return (vision_dim*llm_dim + llm_dim) + (llm_dim*llm_dim + llm_dim)
def dynamic_tokens(h, w, patch=14, compress=4, max_tokens=2560):
    while True:
        raw = math.ceil(h/patch) * math.ceil(w/patch)
        tokens = raw // compress
        if tokens <= max_tokens or h < patch or w < patch:
            return int(tokens)
        h, w = int(h*0.8), int(w*0.8)
import torch
import torch.nn.functional as F
