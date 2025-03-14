from typing import Literal, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.inference_mode()
def search_for_w(
    M: torch.Tensor,
    C: torch.Tensor,
    r: float,
    *,
    initial_w: float,
    max_iter: int,
    abs_tol: float,
    max_w: float = 1e8,
) -> tuple[torch.Tensor, float]:
    """
        Search for a w such that 
            ((M + w * C) <= torch.diag(M)).mean() is approximately r

        Args:
        - M: nonnegative cost torch.Tensor of shape (N, N)
        - C: nonnegative cost torch.Tensor of shape (N, N)
        - r: target value
        - initial_w: initial guess for w
        - max_iter: maximum number of iterations
        - abs_tol: absolute tolerance for r
        - max_w: maximum value for w

        Returns:
        - (M + w * C): torch.Tensor of shape (N, N)
        - w: float
        """

    low = 0
    high = initial_w

    M_diag = torch.diag(M)

    def r_fn(w: float) -> tuple[torch.Tensor, float]:
        cost = (M + w * C)
        return cost, (cost <= M_diag).float().mean()

    # r is maximized when w is 0
    cost, curr_r = r_fn(0)
    if curr_r < r:
        return cost, 0

    # exponential search for high
    for _ in range(max_iter):
        _, curr_r = r_fn(high)
        if curr_r > r:
            low = high
            high *= 2
        else:
            break
        if high > max_w:
            return (M + max_w * C), max_w

    # binary search
    for _ in range(max_iter):
        mid = (low + high) / 2
        cost, curr_r = r_fn(mid)
        if curr_r < r:
            high = mid
        else:
            low = mid
        if abs(curr_r - r) < abs_tol:
            return cost, mid

    return cost, mid


class OTConditionalPlanSampler:

    def __init__(
        self,
        condition_weight: Optional[float] = None,
        target_r: Optional[float] = None,
        max_num_iter: int = 10,
        target_r_tol: float = 1e-3,
        x_norm: Literal['l2_squared', 'l2'] = 'l2_squared',
        condition_norm: Literal['l2_squared', 'cosine'] = 'cosine',
    ):
        """
            Args:
            - condition_weight: weight for the condition cost matrix, aka w
            - target_r: target value for r
            - max_num_iter: maximum number of iterations for the search
            - target_r_tol: absolute tolerance for r
            - x_norm: norm for the x cost matrix
            - condition_norm: norm for the condition cost matrix
        """
        self.condition_weight = condition_weight
        self.target_r = target_r
        self.max_num_iter = max_num_iter
        self.target_r_tol = target_r_tol
        self.x_norm = x_norm
        self.condition_norm = condition_norm

    def get_x_cost_matrix(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
            Compute the pairwise distance matrix between x0 and x1

            Args:
            - x0: torch.Tensor of shape (N, ...)
            - x1: torch.Tensor of shape (N, ...)

            Returns:
            - M: torch.Tensor of shape (N, N)
        """
        x0 = x0.view(x0.shape[0], -1)
        x1 = x1.view(x1.shape[0], -1)

        if self.x_norm == 'l2_squared':
            M = torch.cdist(x0, x1)**2
        elif self.x_norm == 'l2':
            M = torch.cdist(x0, x1)
        else:
            raise ValueError(f'Unknown x norm: {self.x_norm}')

        return M

    def get_condition_cost_matrix(self, y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
        """
            Compute the pairwise distance matrix between y0 and y1

            Args:
            - y0: torch.Tensor of shape (N, ...)
            - y1: torch.Tensor of shape (N, ...)

            Returns:
            - C: torch.Tensor of shape (N, N)
        """
        y0 = y0.view(y1.shape[0], -1)
        y1 = y1.view(y1.shape[0], -1)

        if self.condition_norm == 'cosine':
            y0 /= y0.norm(dim=-1, keepdim=True)
            y1 /= y1.norm(dim=-1, keepdim=True)
            C = 1 - torch.mm(y0, y1.transpose(0, 1))
        elif self.condition_norm == 'l2_squared':
            C = torch.cdist(y0, y1)**2
        else:
            raise ValueError(f'Unknown condition norm: {self.condition_norm}')

        return C

    @torch.inference_mode()
    def sample_plan_with_labels(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0: Optional[torch.Tensor],
        y1: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            Given prior x0, data x1, prior condition y0, and data condition y1 (usually y0==y1), 
            returns the coupled data points (x0, x1) and (y0, y1)

            Args:
            - x0: torch.Tensor of shape (N, ...)
            - x1: torch.Tensor of shape (N, ...)
            - y0: Optional, torch.Tensor of shape (N, ...)
            - y1: Optional, torch.Tensor of shape (N, ...)

            Returns:
            - x0: torch.Tensor of shape (N, ...)
            - x1: torch.Tensor of shape (N, ...)
            - y0: Optional, torch.Tensor of shape (N, ...)
            - y1: Optional, torch.Tensor of shape (N, ...)
        """

        M = self.get_x_cost_matrix(x0, x1)

        if self.condition_weight:
            assert y0 is not None and y1 is not None, 'y0 and y1 must be provided if condition_weight is not None'
            _y0 = y0.type_as(x0)
            _y1 = y1.type_as(x1)

            C = self.get_condition_cost_matrix(_y0, _y1)

            if self.target_r is not None:
                cost, self.condition_weight = search_for_w(M,
                                                           C,
                                                           self.target_r,
                                                           initial_w=self.condition_weight,
                                                           max_iter=self.max_num_iter,
                                                           abs_tol=self.target_r_tol)
            else:
                cost = M + C * self.condition_weight
        else:
            cost = M

        i, j = linear_sum_assignment(cost.cpu())

        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )
