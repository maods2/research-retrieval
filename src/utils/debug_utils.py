import torch
import numpy as np

from tqdm import tqdm


def print_grad_stats(model: torch.nn.Module, progress_bar: tqdm = None, **extra_kwargs): 
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorm = float(param.grad.detach().norm(2).item())
            grad_norms.append(gnorm)
        else:
            gnorm = 0.0
        # optional: log per-parameter norm
        # if progress_bar is not None:
        #     progress_bar.set_postfix({f'grad_norm_{name}': gnorm})
        # else:
        #     print(f"{name}: {gnorm:.6f}")

    if grad_norms:
        avg_grad = float(np.mean(grad_norms))
        max_grad = float(np.max(grad_norms))
    else:
        avg_grad = 0.0
        max_grad = 0.0

    if progress_bar is not None:
        progress_bar.set_postfix(
            {'avg_grad_norm': avg_grad, 'max_grad_norm': max_grad}
            | extra_kwargs
        )
    else:
        print(f"Average gradient norm: {avg_grad:.6f}")
        print(f"Maximum gradient norm: {max_grad:.6f}")
