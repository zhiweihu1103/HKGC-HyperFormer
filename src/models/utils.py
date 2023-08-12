import torch
import numpy as np

from torch.nn import Parameter
from torch.nn.init import xavier_normal_

def save_model(config: dict, model: torch.nn.Module, model_path: str):
    torch.save({'config': config, 'model': model.state_dict()}, model_path)

def score2str(score):
    loss = score['loss']
    hits1 = score['hits@1']
    hits3 = score['hits@3']
    hits10 = score['hits@10']
    mrr = score['MRR']
    return f'loss: {loss}, hits@1: {hits1}, hits@3: {hits3}, hits@10: {hits10}, MRR: {mrr}'

def get_ranks(probs, labels, filters=None):
    log_probs = torch.log(probs)
    if filters is not None:
        log_probs[filters[:, 0], filters[:, 1]] = -np.inf
    sorted_idx = torch.argsort(log_probs, dim=-1, descending=True)
    labels = labels.unsqueeze(dim=-1)
    rank = torch.nonzero(torch.eq(sorted_idx, labels))[:, 1] + 1
    return rank.cpu().numpy().tolist()

def get_norms(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def get_scores(rank: list, loss=None):
    rank = np.array(rank)
    hits1 = round(np.mean(rank <= 1) * 100, 2)
    hits3 = round(np.mean(rank <= 3) * 100, 2)
    hits10 = round(np.mean(rank <= 10) * 100, 2)
    mrr = round(float(np.mean(1. / rank)), 4)
    loss = round(loss, 2)
    return {'loss': loss, 'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10, 'MRR': mrr}

def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param

def conj(a):
    a[..., 1] = -a[..., 1]
    return a

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def rotate(h, r):
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)
