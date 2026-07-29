import logging
import os
import math
import torch 
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F

_TRACE_STEPS_LEFT: int | None = None


def _consume_trace_budget() -> bool:
    global _TRACE_STEPS_LEFT
    if _TRACE_STEPS_LEFT is None:
        raw = os.environ.get("MINLSTM_TRACE_STEPS", "0").strip().lower()
        if raw in {"true", "yes", "on"}:
            _TRACE_STEPS_LEFT = 1
        else:
            try:
                _TRACE_STEPS_LEFT = max(0, int(raw))
            except ValueError:
                _TRACE_STEPS_LEFT = 0
    if _TRACE_STEPS_LEFT <= 0:
        return False
    _TRACE_STEPS_LEFT -= 1
    return True



def g(x: torch.Tensor) -> torch.Tensor:
    """
    Positive-valued transform used in the paper so that:
        h_tilde = g(W_h x_t)
    is strictly positive, which allows us to take log(h_tilde).

    Paper definition:
        g(x) = x + 0.5     if x >= 0
             = sigmoid(x)  otherwise

    This guarantees:
        g(x) > 0
    """
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


# parallel scan 

def parallel_scan_log(
    log_coeffs: torch.Tensor,
    log_values: torch.Tensor,
) -> torch.Tensor:
    """
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    Compute h_1, ..., h_T for the recurrence:

        h_t = a_t ⊙ h_{t-1} + b_t
    
    given:
        log_coeffs = log(a_1), ..., log(a_T)
        log_values = log(b_0), log(b_1), ..., log(b_T)
    
    Shapes:
        log_coeffs: (B, T, H)
        log_values: (B, T+1, H)
    
    Returns:
        h: (B, T, H)
    
    # Derivation for sanity 

    Let
        A_t = prod_{k=1}^t a_k

    Then the recurrence can be expanded as:
        h_t = A_t * ( h_0 + sum_{k=1}^t b_k / A_k )

    In log-space:
        log A_t = sum_{k=1}^t log a_k
    
    The implementation computes:
        a_star[t] = log A_t
    
    and then uses logcumsumexp to stably compute:
        log( h_0 + sum_{k=1}^t b_k / A_k )
    
    """
    # a_star[t] = log A_t = sum_{k=1}^t log a_k
    # Shape before pad: (B, T, H)
    # After pad:        (B, T+1, H)
     # F.pad(..., (0, 0, 1, 0)) means:
    #   - no padding on last dimension (H)
    #   - prepend 1 zero on time dimension
    #
    # So:
    #   a_star[:, 0] = 0  = log A_0
    #   a_star[:, 1] = log a_1
    #   a_star[:, 2] = log(a_1 a_2)
    #   ...

    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))

    # Compute:
    #   log( h_0 + sum_{k=1}^t b_k / A_k )
    #
    # Since:
    #   log(b_k / A_k) = log b_k - log A_k
    #
    # log_values contains [log b_0, log b_1, ..., log b_T]
    # and a_star contains    [log A_0, log A_1, ..., log A_T]
    #
    # Therefore:
    #   log_values - a_star = log( b_k / A_k )
    #
    # logcumsumexp then performs cumulative summation in log-space.
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)

    # Recover:
    #   log h_t = log A_t + log( h_0 + sum_{k=1}^t b_k / A_k )
    log_h = a_star + log_h0_plus_b_star

    # Drop h_0 and return h_1 ... h_T
    return torch.exp(log_h)[:, 1:]


def log_g(x: torch.Tensor) -> torch.Tensor:
    """
    Log of g(x), written in a numerically stable way.

    If x >= 0:
        g(x) = x + 0.5
        log g(x) = log(x + 0.5)

    Else:
        g(x) = sigmoid(x)
        log g(x) = log(sigmoid(x)) = -softplus(-x)
    """
    return torch.where(
        x >= 0,
        (F.relu(x) + 0.5).log(),
        -F.softplus(-x)
    )
# ============================================================
# Full minLSTM layer
# ============================================================

class MinLSTMLayer(nn.Module):
    """
    minLSTM layer with:
    - parallel log-space forward for training
    - sequential forward for inference

    Input:
        x: (B, T, input_size)

    Output:
        h: (B, T, hidden_size)

    Core recurrence:
        h_t = f'_t ⊙ h_{t-1} + i'_t ⊙ h_tilde_t

    with:
        f_t = sigmoid(W_f x_t + b_f)
        i_t = sigmoid(W_i x_t + b_i)

        f'_t = f_t / (f_t + i_t)
        i'_t = i_t / (f_t + i_t)

        h_tilde_t = g(W_h x_t + b_h)

    For parallel training, we identify:
        a_t = f'_t
        b_t = i'_t ⊙ h_tilde_t

    so:
        h_t = a_t ⊙ h_{t-1} + b_t
    and solve it with parallel_scan_log.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        learnable_h0: bool = False,
        h0_init: float = 1.0,
        forget_bias_init: Optional[float] = None,
        candidate_log_shift: float = 0.0,
    ):
        super().__init__()
        if h0_init <= 0:
            raise ValueError("h0_init must be > 0 because log(h0) is used in parallel mode.")

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)
        if forget_bias_init is not None:
            nn.init.constant_(self.linear_f.bias, float(forget_bias_init))
        self.candidate_log_shift = float(candidate_log_shift)
        self.candidate_scale = math.exp(self.candidate_log_shift)
        self.debug_every = max(1, int(os.getenv("MINLSTM_DEBUG_EVERY", "200")))
        self._gate_debug_step = 0

        h0 = torch.full((1, 1, hidden_size), float(h0_init))
        if learnable_h0:
            self.h0 = nn.Parameter(h0)
        else:
            self.register_buffer("h0", h0)

    def _get_h0(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Return initial hidden state h_0 with shape (B, 1, H).
        Must remain positive for log-space forward.
        """
        h0 = self.h0.to(device=device, dtype=dtype)
        return h0.expand(batch_size, 1, self.hidden_size)

    @staticmethod
    def _log_gate_stats(f_prime_t: torch.Tensor, i_prime_t: torch.Tensor, mode: str) -> None:
        """Diagnostics for gate behavior and collapse checks."""
        f_stats = f_prime_t.detach()
        i_stats = i_prime_t.detach()
        f_mean = f_stats.mean().item()
        i_mean = i_stats.mean().item()
        f_std = f_stats.std().item()
        i_std = i_stats.std().item()
        logging.info(
            "[minLSTM:%s] gate stats | f_mean=%.6f f_std=%.6f f_min=%.6f f_max=%.6f i_mean=%.6f i_std=%.6f",
            mode,
            f_mean,
            f_std,
            f_stats.min().item(),
            f_stats.max().item(),
            i_mean,
            i_std,
        )
        gate_collapse = (f_std < 1e-3) or (i_std < 1e-3) or (f_mean < 0.02) or (f_mean > 0.98)
        if gate_collapse:
            logging.warning(
                "[minLSTM:%s] gate collapse suspect | f_mean=%.6f f_std=%.6f f_min=%.6f f_max=%.6f i_mean=%.6f i_std=%.6f",
                mode,
                f_mean,
                f_std,
                f_stats.min().item(),
                f_stats.max().item(),
                i_mean,
                i_std,
            )

    def _should_log_gate_stats(self) -> bool:
        self._gate_debug_step += 1
        return (self._gate_debug_step % self.debug_every) == 0

    def forward_sequential(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sequential inference mode.

        Args:
            x:   (B, T, input_size)
            h_0: (B, 1, H) or None

        Returns:
            h:   (B, T, H)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, T, input_size), got {x.shape}")

        B, T, _ = x.shape
        if h_0 is None:
            h_0 = self._get_h0(B, x.device, x.dtype)

        if h_0.shape != (B, 1, self.hidden_size):
            raise ValueError(
                f"h_0 must have shape {(B, 1, self.hidden_size)}, got {h_0.shape}"
            )

        h_prev = h_0[:, 0, :]  # (B, H)
        outputs = []

        for t in range(T):
            x_t = x[:, t, :]  # (B, input_size)

            # f_t = sigmoid(W_f x_t + b_f)
            f_t = torch.sigmoid(self.linear_f(x_t))

            # i_t = sigmoid(W_i x_t + b_i)
            i_t = torch.sigmoid(self.linear_i(x_t))

            # f'_t = f_t / (f_t + i_t)
            # i'_t = i_t / (f_t + i_t)
            denom = f_t + i_t + 1e-8
            f_prime_t = f_t / denom
            i_prime_t = i_t / denom

            # h_tilde_t = g(W_h x_t + b_h)
            h_tilde_t = g(self.linear_h(x_t)) * self.candidate_scale

            # h_t = f'_t ⊙ h_{t-1} + i'_t ⊙ h_tilde_t
            h_t = f_prime_t * h_prev + i_prime_t * h_tilde_t
            
            outputs.append(h_t.unsqueeze(1))
            h_prev = h_t

        if self._should_log_gate_stats():
            self._log_gate_stats(f_prime_t, i_prime_t, mode="sequential")

        return torch.cat(outputs, dim=1)

    def forward_parallel(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parallel training mode using log-space scan.

        Args:
            x:   (B, T, input_size)
            h_0: (B, 1, H) or None, must be strictly positive

        Returns:
            h:   (B, T, H)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, T, input_size), got {x.shape}")

        B, T, _ = x.shape
        if h_0 is None:
            h_0 = self._get_h0(B, x.device, x.dtype)

        if h_0.shape != (B, 1, self.hidden_size):
            raise ValueError(
                f"h_0 must have shape {(B, 1, self.hidden_size)}, got {h_0.shape}"
            )
        if torch.any(h_0 <= 0):
            raise ValueError("h_0 must be strictly positive for log-space forward.")

        # --------------------------------------------------------
        # Compute log(f'_t) and log(i'_t) in a numerically stable way
        #
        # Let:
        #   p_t = W_f x_t + b_f
        #   k_t = W_i x_t + b_i
        #
        # Then:
        #   f_t = sigmoid(p_t)
        #   i_t = sigmoid(k_t)
        #
        # We need:
        #   log f'_t = log( f_t / (f_t + i_t) )
        #   log i'_t = log( i_t / (f_t + i_t) )
        #
        # Using:
        #   log(sigmoid(x)) = -softplus(-x)
        #
        # We get:
        #   diff = softplus(-p_t) - softplus(-k_t)
        #   log f'_t = -softplus(diff)
        #   log i'_t = -softplus(-diff)
        # --------------------------------------------------------
        p = self.linear_f(x)  # (B, T, H)
        k = self.linear_i(x)  # (B, T, H)

        diff = F.softplus(-p) - F.softplus(-k)

        # log f'_t
        log_f = -F.softplus(diff)

        # log i'_t
        log_i = -F.softplus(-diff)

        if self._should_log_gate_stats():
            self._log_gate_stats(torch.exp(log_f), torch.exp(log_i), mode="parallel")

        # --------------------------------------------------------
        # Candidate:
        #   h_tilde_t = g(W_h x_t + b_h)
        # so:
        #   log h_tilde_t = log_g(W_h x_t + b_h)
        # --------------------------------------------------------
        log_tilde_h = log_g(self.linear_h(x)) + self.candidate_log_shift

        # --------------------------------------------------------
        # Recurrence:
        #   h_t = a_t ⊙ h_{t-1} + b_t
        #
        # with:
        #   a_t = f'_t
        #   b_t = i'_t ⊙ h_tilde_t
        #
        # Therefore:
        #   log a_t = log_f
        #   log b_t = log_i + log_tilde_h
        #
        # The scan expects:
        #   log_values = [log b_0, log b_1, ..., log b_T]
        #
        # where b_0 is the initial hidden state h_0.
        # --------------------------------------------------------



        log_h_0 = torch.log(h_0)
        log_b = log_i + log_tilde_h

        h = parallel_scan_log(
            log_coeffs=log_f,
            log_values=torch.cat([log_h_0, log_b], dim=1),
        )
        return h

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        parallel: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Main forward.

        If parallel is None:
            - uses parallel mode during training
            - uses sequential mode during eval
        """
        if parallel is None:
            parallel = self.training

        if _consume_trace_budget():
            logging.info(
                "[TRACE minLSTM] dispatch training=%s parallel=%s input=%s hidden=%d",
                self.training,
                parallel,
                tuple(x.shape),
                self.hidden_size,
            )

        if parallel:
            return self.forward_parallel(x, h_0)
        return self.forward_sequential(x, h_0)


class HierarchicalMinLSTMPooling(nn.Module):

    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        forget_bias_init: Optional[float] = None,
        candidate_log_shift: float = 0.0,
    ):

        super().__init__()
        #level 1 compression 
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=4)
        # Normalize each frame vector before the recurrent update.
        self.input_norm1 = nn.LayerNorm(input_channels)
        self.minlstm1 = MinLSTMLayer(
            input_size=input_channels,
            hidden_size=hidden_size,
            forget_bias_init=forget_bias_init,
            candidate_log_shift=candidate_log_shift,
        )

        #level 2 compression 
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
        # Stage 2 sees hidden-state frames, so normalize across hidden dim.
        self.input_norm2 = nn.LayerNorm(hidden_size)
        self.minlstm2 = MinLSTMLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            forget_bias_init=forget_bias_init,
            candidate_log_shift=candidate_log_shift,
        )
        self._pool_debug_step = 0
    
    def forward(self, x, parallel=None):

        # x: (B, C, T)
        if x.dim() != 3:
            raise ValueError(f"x must be (B, C, T), got {x.shape}")
        seq_len_in = x.shape[2]
        trace = _consume_trace_budget()
        if trace:
            logging.info("[TRACE hier-minlstm] input frame_x (B,C,T): %s", tuple(x.shape))

        # ---------- Level 1 ----------
        # temporal compression
        x = self.pool1(x)              # (B, C, T/4)
        seq_len_l1 = x.shape[2]
        if trace:
            logging.info("[TRACE hier-minlstm] after pool1 (B,C,T/4): %s", tuple(x.shape))

        # convert to sequence for RNN
        x = x.transpose(1, 2)          # (B, T/4, C)
        if trace:
            logging.info("[TRACE hier-minlstm] stage1 transpose -> (B,T/4,C): %s", tuple(x.shape))
        x = self.input_norm1(x)
        if trace:
            logging.info("[TRACE hier-minlstm] stage1 layernorm -> (B,T/4,C): %s", tuple(x.shape))

        # run first minLSTM
        h1 = self.minlstm1(x, parallel=parallel)   # (B, T/4, H)
        if trace:
            logging.info("[TRACE hier-minlstm] stage1 h1 (B,T/4,H): %s", tuple(h1.shape))

        # ---------- Level 2 ----------
        # convert back for pooling
        h1 = h1.transpose(1, 2)        # (B, H, T/4)
        if trace:
            logging.info("[TRACE hier-minlstm] stage2 pre-pool transpose -> (B,H,T/4): %s", tuple(h1.shape))

        # compress again
        h1 = self.pool2(h1)            # (B, H, T/16)
        seq_len_l2 = h1.shape[2]
        if trace:
            logging.info("[TRACE hier-minlstm] after pool2 (B,H,T/16): %s", tuple(h1.shape))

        # back to RNN format
        h1 = h1.transpose(1, 2)        # (B, T/16, H)
        if trace:
            logging.info("[TRACE hier-minlstm] stage2 transpose -> (B,T/16,H): %s", tuple(h1.shape))
        h1 = self.input_norm2(h1)
        if trace:
            logging.info("[TRACE hier-minlstm] stage2 layernorm -> (B,T/16,H): %s", tuple(h1.shape))

        # second minLSTM
        h2 = self.minlstm2(h1, parallel=parallel)  # (B, T/16, H)
        if trace:
            logging.info("[TRACE hier-minlstm] stage2 h2 (B,T/16,H): %s", tuple(h2.shape))

        # ---------- Speaker representation ----------
        # take final hidden state
        pooled = h2[:, -1, :]          # (B, H)
        if trace:
            logging.info("[TRACE hier-minlstm] pooled h_T (B,H): %s", tuple(pooled.shape))

        # Keep periodic diagnostics aligned with MinLSTMPooling logs.
        h1_detached = h1.detach()
        h2_detached = h2.detach()
        pooled_detached = pooled.detach()
        hidden_time_std_l1 = h1_detached.std(dim=1).mean().item()
        hidden_time_std_l2 = h2_detached.std(dim=1).mean().item()
        embedding_batch_std = pooled_detached.std(dim=0).mean().item()
        self._pool_debug_step += 1
        if (self._pool_debug_step % self.minlstm1.debug_every) == 0:
            logging.info(
                "[hier-minLSTM] periodic stats | hidden_time_std_l1=%.6f hidden_time_std_l2=%.6f embedding_batch_std=%.6f sequence_length_in=%d sequence_length_l1=%d sequence_length_l2=%d",
                hidden_time_std_l1,
                hidden_time_std_l2,
                embedding_batch_std,
                seq_len_in,
                seq_len_l1,
                seq_len_l2,
            )
        if (
            (hidden_time_std_l1 < 1e-3)
            or (hidden_time_std_l2 < 1e-3)
            or (embedding_batch_std < 1e-3)
        ):
            logging.warning(
                "[hier-minLSTM] collapse suspect | hidden_time_std_l1=%.6f hidden_time_std_l2=%.6f embedding_batch_std=%.6f sequence_length_in=%d sequence_length_l1=%d sequence_length_l2=%d",
                hidden_time_std_l1,
                hidden_time_std_l2,
                embedding_batch_std,
                seq_len_in,
                seq_len_l1,
                seq_len_l2,
            )

        return pooled

            
                  
class MinLSTMPooling(nn.Module):
    """
    Convenience wrapper for ECAPA-style use.

    Input:
        x: (B, C, T)

    Output:
        pooled: (B, hidden_size)

    This wrapper:
    1. transposes to (B, T, C)
    2. runs minLSTM over time
    3. returns the last state h_T

    This is the direct replacement for a temporal pooling layer.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        forget_bias_init: Optional[float] = None,
        candidate_log_shift: float = 0.0,
    ):
        super().__init__()
        # Normalize each frame vector before minLSTM so the recurrence sees
        # standardized per-frame channel activations.
        self.input_norm = nn.LayerNorm(input_channels)
        self.minlstm = MinLSTMLayer(
            input_size=input_channels,
            hidden_size=hidden_size,
            learnable_h0=False,
            h0_init=1.0,
            forget_bias_init=forget_bias_init,
            candidate_log_shift=candidate_log_shift,
        )
        self._pool_debug_step = 0
        

    def forward(self, x: torch.Tensor, parallel: Optional[bool] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            pooled: (B, hidden_size)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, C, T), got {x.shape}")
        

        seq_len = x.shape[2]
        trace = _consume_trace_budget()
        if trace:
            logging.info("[TRACE minlstm] input frame_x (B,C,T): %s", tuple(x.shape))

        # Convert ECAPA frame tensor from (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        if trace:
            logging.info("[TRACE minlstm] transpose -> (B,T,C): %s", tuple(x.shape))
        x = self.input_norm(x)
        if trace:
            logging.info("[TRACE minlstm] layernorm -> (B,T,C): %s", tuple(x.shape))

        # h: (B, T, hidden_size)
        h = self.minlstm(x, parallel=parallel)
        if trace:
            logging.info("[TRACE minlstm] minlstm hidden sequence h: %s", tuple(h.shape))

        # final state h_T
        pooled = h[:, -1, :]
        if trace:
            logging.info("[TRACE minlstm] pooled h_T (B,H): %s", tuple(pooled.shape))

        h_detached = h.detach()
        pooled_detached = pooled.detach()
        hidden_time_std = h_detached.std(dim=1).mean().item()
        embedding_batch_std = pooled_detached.std(dim=0).mean().item()
        self._pool_debug_step += 1
        if (self._pool_debug_step % self.minlstm.debug_every) == 0:
            logging.info(
                "[minLSTM] periodic stats | hidden_time_std=%.6f embedding_batch_std=%.6f sequence_length=%d",
                hidden_time_std,
                embedding_batch_std,
                seq_len,
            )
        if (hidden_time_std < 1e-3) or (embedding_batch_std < 1e-3):
            logging.warning(
                "[minLSTM] collapse suspect | hidden_time_std=%.6f embedding_batch_std=%.6f sequence_length=%d",
                hidden_time_std,
                embedding_batch_std,
                seq_len,
            )

        return pooled


class MinLSTMAttentiveStatsPooling(nn.Module):
    """
    minLSTM followed by attentive statistics pooling over the full hidden-state sequence.

    Input:
        x: (B, C, T)

    Output:
        pooled: (B, hidden_size)

    This wrapper:
    1. transposes to (B, T, C)
    2. runs minLSTM over time
    3. applies attentive stats pooling over all hidden states
    4. projects pooled (B, 2H) back to (B, H) for head compatibility
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        forget_bias_init: Optional[float] = None,
        candidate_log_shift: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(input_channels)
        self.minlstm = MinLSTMLayer(
            input_size=input_channels,
            hidden_size=hidden_size,
            learnable_h0=False,
            h0_init=1.0,
            forget_bias_init=forget_bias_init,
            candidate_log_shift=candidate_log_shift,
        )
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_size * 3, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, hidden_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.stats_proj = nn.Linear(hidden_size * 2, hidden_size)
        self._pool_debug_step = 0

    def forward(self, x: torch.Tensor, parallel: Optional[bool] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be (B, C, T), got {x.shape}")

        seq_len = x.shape[2]
        trace = _consume_trace_budget()
        if trace:
            logging.info("[TRACE minlstm-asp] input frame_x (B,C,T): %s", tuple(x.shape))

        x = x.transpose(1, 2)
        if trace:
            logging.info("[TRACE minlstm-asp] transpose -> (B,T,C): %s", tuple(x.shape))
        x = self.input_norm(x)
        if trace:
            logging.info("[TRACE minlstm-asp] layernorm -> (B,T,C): %s", tuple(x.shape))

        h = self.minlstm(x, parallel=parallel)
        if trace:
            logging.info("[TRACE minlstm-asp] minlstm hidden sequence h: %s", tuple(h.shape))

        hidden_seq = h.transpose(1, 2)
        t = hidden_seq.size(-1)
        global_h = torch.cat(
            (
                hidden_seq,
                torch.mean(hidden_seq, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(hidden_seq, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )
        if trace:
            logging.info("[TRACE minlstm-asp] global_h (B,3H,T): %s", tuple(global_h.shape))
        w = self.attention(global_h)
        mu = torch.sum(hidden_seq * w, dim=2)
        sg = torch.sqrt((torch.sum((hidden_seq ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        pooled_stats = torch.cat((mu, sg), dim=1)
        if trace:
            logging.info("[TRACE minlstm-asp] pooled attentive stats (B,2H): %s", tuple(pooled_stats.shape))
        pooled = self.stats_proj(pooled_stats)
        if trace:
            logging.info("[TRACE minlstm-asp] projected pooled (B,H): %s", tuple(pooled.shape))

        h_detached = h.detach()
        pooled_detached = pooled.detach()
        hidden_time_std = h_detached.std(dim=1).mean().item()
        embedding_batch_std = pooled_detached.std(dim=0).mean().item()
        self._pool_debug_step += 1
        if (self._pool_debug_step % self.minlstm.debug_every) == 0:
            logging.info(
                "[minLSTM-ASP] periodic stats | hidden_time_std=%.6f embedding_batch_std=%.6f sequence_length=%d",
                hidden_time_std,
                embedding_batch_std,
                seq_len,
            )
        if (hidden_time_std < 1e-3) or (embedding_batch_std < 1e-3):
            logging.warning(
                "[minLSTM-ASP] collapse suspect | hidden_time_std=%.6f embedding_batch_std=%.6f sequence_length=%d",
                hidden_time_std,
                embedding_batch_std,
                seq_len,
            )

        return pooled


def _test_ecapa_pooling_wrapper():
    torch.manual_seed(0)
    B, C, T = 2, 1536, 50 
    hidden = 32
    pool = MinLSTMPooling(input_channels=C, hidden_size=hidden)
    x = torch.randn(B, C, T)
    y = pool(x, parallel=True)
    assert y.shape == (B, hidden)
    print("✓ ECAPA pooling wrapper test passed")

if __name__ == "__main__":
    _test_ecapa_pooling_wrapper()
