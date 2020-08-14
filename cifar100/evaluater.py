from __future__ import print_function, absolute_import
import torch.nn.functional as F
__all__ = ['top_error', 'nll']

def top_error(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100 - correct_k.mul_(100.0 / batch_size).detach().cpu().item())
    return res

def nll(output, target):
    res = F.nll_loss(F.log_softmax(output, dim=1), target).detach().cpu().item()
    return res  