# ~ similar to https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
import torch
from torch.optim import Optimizer


class SGDModified(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDModified, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDModified, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, grads, mode, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i,p in enumerate(group['params']):
                if mode == 'normal' or mode=='draco_lite' or mode=='draco_lite_attack' or mode=='byzshield_cpu':
                    d_p = torch.from_numpy(grads[i]).float()
                elif mode == 'byzshield_gpu':
                    # # ~ For this, I use the same logic as in util.py as default device for .cuda() is "cuda:0"
                    # if torch.cuda.device_count() > 1:
                        # # ~ Use 2nd GPU
                        # torch.cuda.set_device(torch.device("cuda:1"))
                    # else:
                        # # ~ Use 1st GPU
                        # torch.cuda.set_device(torch.device("cuda:0"))
                
                    # ~ This is only for BYZSHIELD and ASPIS where CUDA is supported
                    d_p = torch.from_numpy(grads[i]).cuda().float()
                    
                #elif mode=='geometric_median' or mode=='maj_vote' or mode=='cyclic' or mode=='krum':
                elif mode in ('geometric_median', 'maj_vote', 'cyclic', 'krum', 'multi-krum', 'bulyan', 'coord-median', 'sign-sgd'):
                    d_p = torch.from_numpy(grads[i].reshape(p.size())).float()
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        return loss

    def lr_update(self, updated_lr):
        for group in self.param_groups:
            group['lr'] = updated_lr