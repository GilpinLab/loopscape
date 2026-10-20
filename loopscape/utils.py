import os
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def batched_apply(fn, x, batch_size=1024, dim=0, max_workers=1, pad_dim=None):
    """
    Apply a function to a large tensor in chunks and concatenate the outputs.

    Args:
        fn (callable): Function mapping a tensor chunk to an output tensor.
        x (torch.Tensor): Input tensor, e.g. shape (B, 97, 512).
        batch_size (int, optional): Number of items per chunk along `dim`.
        dim (int, optional): Batch dimension to split and concatenate along.
        max_workers (int, optional): Number of worker threads. if > 1, uses
            ThreadPoolExecutor for parallel processing.
        pad_dim (int, optional): If set, chunk outputs whose length along this
            dimension differs (e.g. solver step counts under early halting) are
            padded to the longest chunk by repeating their final slice before
            concatenation.

    Returns:
        (torch.Tensor): Concatenated output, e.g. shape (B, 16, 9, 9).
    """
    if max_workers == -1:
        max_workers = os.cpu_count() or 1
    with torch.inference_mode():
        chunks = torch.split(x, batch_size, dim=dim)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                ys = list(ex.map(fn, chunks))
        else:
            ys = [fn(chunk) for chunk in chunks]

        if pad_dim is not None:
            target = max(y.shape[pad_dim] for y in ys)

            def _pad(y):
                n = target - y.shape[pad_dim]
                if n == 0:
                    return y
                last = y.narrow(pad_dim, y.shape[pad_dim] - 1, 1)
                reps = [1] * y.ndim
                reps[pad_dim] = n
                return torch.cat([y, last.repeat(reps)], dim=pad_dim)

            ys = [_pad(y) for y in ys]

        return torch.cat(ys, dim=dim)


import gzip, io

def load_pt_gz(path, device=None, weights_only=False):
    with gzip.open(path, "rb") as fh:
        return torch.load(fh, map_location=device, weights_only=weights_only)