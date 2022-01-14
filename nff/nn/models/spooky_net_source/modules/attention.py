import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional


class Attention(nn.Module):
    """
    Efficient (linear scaling) approximation for attention described in
    Choromanski, K., et al. "Rethinking Attention with Performers.".

    Arguments:
        dim_qk (int):
            Dimension of query/key vectors.
        dim_v (int):
            Dimension of value vectors.
        num_random_featues (int):
            Number of random features for approximating attention matrix. If
            this is 0, the exact attention matrix is computed.
    """

    def __init__(
        self, dim_qk: int, dim_v: int, num_random_features: Optional[int] = None
    ) -> None:
        """ Initializes the Attention class. """
        super(Attention, self).__init__()
        self.num_random_features = num_random_features
        if self.num_random_features is not None:
            omega = self._omega(num_random_features, dim_qk)
        else:
            omega = []
        self.register_buffer("omega", torch.tensor(omega, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def _omega(self, nrows: int, ncols: int) -> np.ndarray:
        """ Return a (nrows x ncols) random feature matrix. """
        nblocks = int(nrows / ncols)
        blocks = []
        for i in range(nblocks):
            block = np.random.normal(size=(ncols, ncols))
            q, _ = np.linalg.qr(block)
            blocks.append(np.transpose(q))
        missing_rows = nrows - nblocks * ncols
        if missing_rows > 0:
            block = np.random.normal(size=(ncols, ncols))
            q, _ = np.linalg.qr(block)
            blocks.append(np.transpose(q)[:missing_rows])
        norm = np.linalg.norm(  # renormalize rows so they still follow N(0,1)
            np.random.normal(size=(nrows, ncols)), axis=1, keepdims=True
        )
        return (norm * np.vstack(blocks)).T

    def _phi(
        self,
        X: torch.Tensor,
        is_query: bool,
        num_batch: int,
        batch_seg: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """ Normalize X and project into random feature space. """
        d = X.shape[-1]
        m = self.omega.shape[-1]
        U = torch.matmul(X / d ** 0.25, self.omega)
        h = torch.sum(X ** 2, dim=-1, keepdim=True) / (2 * d ** 0.5)  # OLD
        # determine maximum (is subtracted to prevent numerical overflow)
        if is_query:
            maximum, _ = torch.max(U, dim=-1, keepdim=True)
        else:
            if num_batch > 1:
                brow = batch_seg.view(1, -1, 1).expand(num_batch, -1, U.shape[-1])
                bcol = (
                    torch.arange(
                        num_batch, dtype=batch_seg.dtype, device=batch_seg.device
                    )
                    .view(-1, 1, 1)
                    .expand(-1, U.shape[-2], U.shape[-1])
                )
                mask = torch.where(
                    brow == bcol, torch.ones_like(U), torch.zeros_like(U)
                )
                tmp = U.unsqueeze(0).expand(num_batch, -1, -1)
                tmp, _ = torch.max(mask * tmp, dim=-1)
                tmp, _ = torch.max(tmp, dim=-1)
                if tmp.device.type == "cpu":  # indexing faster on CPU
                    maximum = tmp[batch_seg].unsqueeze(-1)
                else:  # gathering is faster on GPUs
                    maximum = torch.gather(tmp, 0, batch_seg).unsqueeze(-1)
            else:
                maximum = torch.max(U)
        return (torch.exp(U - h - maximum) + eps) / math.sqrt(m)

    def _exact_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        eps: float = 1e-8,
    ):
        """ Compute exact attention. """
        d = Q.shape[-1]
        dot = Q @ K.T  # dot product
        A = torch.exp((dot - torch.max(dot)) / d ** 0.5)  # attention matrix
        if num_batch > 1:  # mask out entries of different batches
            brow = batch_seg.view(1, -1).expand(A.shape[-2], -1)
            bcol = batch_seg.view(-1, 1).expand(-1, A.shape[-1])
            mask = torch.where(brow == bcol, torch.ones_like(A), torch.zeros_like(A))
            A = A * mask
        norm = torch.sum(A, -1, keepdim=True) + eps
        return (A / norm) @ V

    def _approximate_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """ Compute approximate attention. """
        Q = self._phi(Q, True, num_batch, batch_seg)  # random projection of Q
        K = self._phi(K, False, num_batch, batch_seg)  # random projection of K
        if num_batch > 1:
            d = Q.shape[-1]
            n = batch_seg.shape[0]

            # compute norm
            idx = batch_seg.unsqueeze(-1).expand(-1, d)
            tmp = K.new_zeros(num_batch, d).scatter_add_(0, idx, K)
            norm = torch.gather(Q @ tmp.T, -1, batch_seg.unsqueeze(-1)) + eps

            # the ops below are equivalent to this loop (but more efficient):
            # return torch.cat([Q[b==batch_seg]@(
            #    K[b==batch_seg].transpose(-1,-2)@V[b==batch_seg])
            #    for b in range(num_batch)])/norm
            if mask is None:  # mask can be shared across multiple attentions
                one_hot = nn.functional.one_hot(batch_seg).to(
                    dtype=V.dtype, device=V.device
                )
                mask = one_hot @ one_hot.transpose(-1, -2)
            return ((mask * (K @ Q.transpose(-1, -2))).transpose(-1, -2) @ V) / norm
        else:
            norm = Q @ torch.sum(K, 0, keepdim=True).T + eps
            return (Q @ (K.T @ V)) / norm

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention for the given query, key and value vectors.
        N: Number of input values.
        dim_qk: Dimension of query/key vectors.
        dim_v: Dimension of value vectors.

        Arguments:
            Q (FloatTensor [N, dim_qk]):
                Matrix of N query vectors.
            K (FloatTensor [N, dim_qk]):
                Matrix of N key vectors.
            V (FloatTensor [N, dim_v]):
                Matrix of N value vectors.
            num_batch (int):
                Number of different batches in the input values.
            batch_seg (LongTensor [N]):
                Index for each input that specifies to which batch it belongs.
                For example, when the input consists of a sequence of size 3 and
                another sequence of size 5, batch_seg would be
                [0, 0, 0, 1, 1, 1, 1, 1] (num_batch would be 2 then).
        Returns:
            y (FloatTensor [N, dim_v]):
                Attention-weighted sum of value vectors.
        """
        if self.num_random_features is None:
            return self._exact_attention(Q, K, V, num_batch, batch_seg)
        else:
            return self._approximate_attention(Q, K, V, num_batch, batch_seg, mask)
