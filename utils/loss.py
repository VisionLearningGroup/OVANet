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


def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.range(0, out_open.size(0) - 1).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open