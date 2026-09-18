"""Sparse connectome propagation with an autograd path for the per-edge gains.

y = W @ x, with W an (N x N) CSR matrix whose values are learnable, x an (N x B)
dense state. Forward and the x-gradient are cuSPARSE spmm; the value-gradient is
sampled_addmm (SDDMM): grad_w_e = sum_b gy[dst_e, b] * x[src_e, b], computed only
on the fixed sparsity pattern, so memory never scales with E x B.
"""
from __future__ import annotations

import torch


class SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, crow, col, crow_t, col_t, perm_t, x):
        n = crow.numel() - 1
        W = torch.sparse_csr_tensor(crow, col, values, size=(n, n))
        y = torch.sparse.mm(W, x)
        ctx.save_for_backward(values, crow, col, crow_t, col_t, perm_t, x)
        return y

    @staticmethod
    def backward(ctx, gy):
        values, crow, col, crow_t, col_t, perm_t, x = ctx.saved_tensors
        n = crow.numel() - 1
        gy = gy.contiguous()
        grad_x = grad_v = None
        if ctx.needs_input_grad[6]:
            Wt = torch.sparse_csr_tensor(crow_t, col_t, values[perm_t], size=(n, n))
            grad_x = torch.sparse.mm(Wt, gy)
        if ctx.needs_input_grad[0]:
            grad_v = _sddmm(crow, col, gy, x)
        return grad_v, None, None, None, None, None, grad_x


def _sddmm(crow, col, gy, x):
    """grad over the sparsity pattern: (gy @ x.T)[dst, src]."""
    n = crow.numel() - 1
    try:
        pattern = torch.sparse_csr_tensor(crow, col, torch.zeros(col.numel(), dtype=gy.dtype, device=gy.device), size=(n, n))
        return torch.sparse.sampled_addmm(pattern, gy, x.t().contiguous()).values()
    except (RuntimeError, NotImplementedError):
        # fallback: chunked gather so peak memory stays bounded
        dst = torch.repeat_interleave(torch.arange(n, device=col.device), crow[1:] - crow[:-1])
        out = torch.empty(col.numel(), dtype=gy.dtype, device=gy.device)
        step = max(1, (1 << 28) // max(1, gy.shape[1]))
        for s in range(0, col.numel(), step):
            e = min(s + step, col.numel())
            out[s:e] = (gy[dst[s:e]] * x[col[s:e]]).sum(1)
        return out


class Connectome(torch.nn.Module):
    """Frozen wiring + frozen signs, learnable positive gain per connection.

    W_ij = sign_ij * exp(theta_ij) * syn_scale_ij where syn_scale is log1p(syn_count)
    (a fixed anatomical prior: more synapses, stronger initial connection).
    """

    def __init__(self, src, dst, sign, syn_count, n: int, init_scale: float = 1.0):
        super().__init__()
        src = torch.as_tensor(src, dtype=torch.long)
        dst = torch.as_tensor(dst, dtype=torch.long)
        sign = torch.as_tensor(sign, dtype=torch.float32)
        syn = torch.as_tensor(syn_count, dtype=torch.float32)
        # CSR over rows = dst (post-synaptic), cols = src (pre-synaptic)
        order = torch.argsort(dst * n + src)
        src, dst, sign, syn = src[order], dst[order], sign[order], syn[order]
        counts = torch.bincount(dst, minlength=n)
        crow = torch.zeros(n + 1, dtype=torch.long)
        crow[1:] = torch.cumsum(counts, 0)
        # transpose pattern (rows = src) with permutation from original edge order
        perm_t = torch.argsort(src * n + dst)
        counts_t = torch.bincount(src, minlength=n)
        crow_t = torch.zeros(n + 1, dtype=torch.long)
        crow_t[1:] = torch.cumsum(counts_t, 0)
        col_t = dst[perm_t]
        self.register_buffer("crow", crow.to(torch.int32) if n < 2**31 else crow)
        self.register_buffer("col", src.to(torch.int32))
        self.register_buffer("crow_t", crow_t.to(torch.int32))
        self.register_buffer("col_t", col_t.to(torch.int32))
        self.register_buffer("perm_t", perm_t)
        self.register_buffer("sign", sign)
        # anatomical prior: normalise by sqrt of in-degree so initial activity is O(1)
        indeg = counts.clamp(min=1).float()
        prior = torch.log1p(syn) / torch.sqrt(indeg[dst]) * init_scale
        self.register_buffer("prior", prior)
        self.log_gain = torch.nn.Parameter(torch.zeros(len(src)))  # theta, one per connection
        self.n = n

    def values(self):
        return self.sign * self.prior * torch.exp(self.log_gain)

    def forward(self, x):  # x: [N, B]
        return SpMM.apply(self.values(), self.crow, self.col, self.crow_t, self.col_t, self.perm_t, x)

    def dense(self):
        """Debug only: dense [N, N] matrix (rows=post, cols=pre)."""
        n = self.n
        W = torch.sparse_csr_tensor(self.crow, self.col, self.values(), size=(n, n))
        return W.to_dense()
