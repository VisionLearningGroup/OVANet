import torch
import torch.nn.functional as F

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en

def pseudo_loss(p, thr=0.9):
    p = F.softmax(p)
    p_max = p.max(1)[0]
    positive = p_max > thr

    loss = -torch.log(p + 1e-5)
    loss = loss.min(1)[0]
    tmp = torch.zeros(loss.size(0)).float().cuda()
    if torch.sum(positive.int()) > 0:
        tmp[positive] = loss[positive]
    return torch.mean(tmp)

def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))

def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)
