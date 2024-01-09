import math
from typing import Callable

import torch


def split(x: torch.Tensor) -> torch.Tensor:
    """
    Take a sequence of inputs that represents a tree level,
    and return all left children and all right children.

    Arguments:
        x (torch.Tensor): shape (B, C, T)

    Returns:
        (torch.Tensor, torch.Tensor): shapes (B, C, T//2), (B, C, T//2)

    >>> split(torch.tensor([1,2,3,4,5,6,7,8])[None, None, :])
    (tensor([[[1, 3, 5, 7]]]), tensor([[[2, 4, 6, 8]]])
    """
    B, C, T = x.size()
    x = x.view(B, C, T//2, 2)
    return x[: , :, :, 0], x[:, :, :, 1]


def merge(lefts: torch.Tensor, rights: torch.Tensor) -> torch.Tensor:
    """
    Take sequences of all left children and sequences of all right children and merge them
    into a single tree level.

    Arguments:
        lefts (torch.Tensor): shape (B, C, T//2)
        rights (torch.Tensor): shape (B, C, T//2)

    Returns:
        (torch.Tensor): shape (B, C, T)

    >>> lefts = torch.tensor([1,3,5,7])[None, None, :]
    >>> rights = torch.tensor([2,4,6,8])[None, None, :]
    >>> merge(lefts, rights)
    tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]])
    """
    B, C, half = lefts.size()
    x = torch.stack([lefts, rights], dim=-1) # (bsz, dim, half, 2)
    return x.view(B, C, half*2)


def scan(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    mul=torch.mul,
    add=torch.add,
    zeros_like=torch.zeros_like
) -> torch.Tensor:
    """Solve a first-order recurrence relation using a reference torch implementation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    Arguments:
        gates (torch.Tensor): shape (B, C, T), must be contiguous.
        tokens (torch.Tensor): shape (B, C, T), must be contiguous.
        mul (callable): multiplication function, defaults to torch.mul
        add (callable): addition function, defaults to torch.add
        zeros_like (callable): function to create a tensor of zeros like the input, defaults to torch.zeros_like

    Returns:
        (torch.Tensor): shape (B, C, T)
    """
    B,C,T = tokens.size()
    level = int(math.log2(T))
    return add(mul(scan1(gates, tokens, mul, add, zeros_like, level=level), gates), tokens)


def scan1(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    mul: Callable,
    add: Callable,
    zeros_like: Callable,
    level: int
):
    left_gates, right_gates = split(gates)
    left_x, right_x = split(tokens)

    # up: sum together
    gates = mul(left_gates, right_gates)
    tokens = add(mul(right_gates, left_x), right_x)

    if level == 1:
        root_x = zeros_like(tokens)
    else:
        root_x = scan1(gates, tokens, mul, add, zeros_like, level=level-1)

    # down: left is root, right is left (+) right
    return merge(root_x, add(mul(root_x, left_gates), left_x))
