'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math
import os
import logging
from typing import Optional

import torch

try:
    import torchaudio
except Exception:
    torchaudio = None
import torch.nn as nn
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


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(
        self,
        C,
        input_dim=80,
        sample_rate=16000,
        use_melspec=True,
        pooling_mode: str = "attentive_stats",
        minlstm_hidden_size: int = 3072,
        minlstm_forget_bias_init: Optional[float] = None,
        minlstm_candidate_log_shift: float = 0.0,
    ):

        super(ECAPA_TDNN, self).__init__()

        self.use_melspec = use_melspec
        self.input_dim = input_dim
        self.sample_rate = sample_rate

        if self.use_melspec:
            if torchaudio is None:
                raise ImportError(
                    "torchaudio is required when use_melspec=True. "
                    "Install it via conda/pip with a build that matches your torch install."
                )
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=512,
                    win_length=400,
                    hop_length=160,
                    f_min=20,
                    f_max=7600,
                    window_fn=torch.hamming_window,
                    n_mels=input_dim,
                ),
            )
            self.specaug = FbankAug()  # Spec augmentation
        else:
            self.torchfbank = None
            self.specaug = None

        if pooling_mode not in {
            "attentive_stats",
            "minlstm",
            "minlstm_asp",
            "hier_minlstm",
            "minlstm_globalproj",
            "hier_minlstm_globalproj",
        }:
            raise ValueError(
                f"Unsupported pooling_mode={pooling_mode}. "
                "Choose from ['attentive_stats', 'minlstm', 'minlstm_asp', "
                "'hier_minlstm', 'minlstm_globalproj', 'hier_minlstm_globalproj']."
            )
        if minlstm_hidden_size <= 0:
            raise ValueError(f"minlstm_hidden_size must be > 0, got {minlstm_hidden_size}")

        self.pooling_mode = pooling_mode
        self.minlstm_hidden_size = int(minlstm_hidden_size)
        self.minlstm_forget_bias_init = minlstm_forget_bias_init
        self.minlstm_candidate_log_shift = float(minlstm_candidate_log_shift)
        self.minlstm_pool = None

        self.conv1 = nn.Conv1d(input_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        # Optional stats-aware projection used only by the *_globalproj minLSTM
        # variants. Keeping it separate preserves old minLSTM reproducibility.
        self.minlstm_global_proj = nn.Conv1d(4608, 256, kernel_size=1)
        pooled_dim = 3072 if self.pooling_mode == "attentive_stats" else self.minlstm_hidden_size
        self.bn5 = nn.BatchNorm1d(pooled_dim)
        self.fc6 = nn.Linear(pooled_dim, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def ensure_pool_initialized(self, device=None, dtype=None, input_channels: int = 1536):
        """Materialize lazy minLSTM pooling modules so checkpoints can load before first forward."""
        if self.pooling_mode == "attentive_stats" or self.minlstm_pool is not None:
            return
        from ..min_lstm.minlstm import (
            HierarchicalMinLSTMPooling,
            MinLSTMAttentiveStatsPooling,
            MinLSTMPooling,
        )

        target_device = device if device is not None else self.fc6.weight.device
        target_dtype = dtype if dtype is not None else self.fc6.weight.dtype
        if self.pooling_mode in {"hier_minlstm", "hier_minlstm_globalproj"}:
            self.minlstm_pool = HierarchicalMinLSTMPooling(
                input_channels=input_channels,
                hidden_size=self.minlstm_hidden_size,
                forget_bias_init=self.minlstm_forget_bias_init,
                candidate_log_shift=self.minlstm_candidate_log_shift,
            ).to(device=target_device, dtype=target_dtype)
        elif self.pooling_mode == "minlstm_asp":
            self.minlstm_pool = MinLSTMAttentiveStatsPooling(
                input_channels=input_channels,
                hidden_size=self.minlstm_hidden_size,
                forget_bias_init=self.minlstm_forget_bias_init,
                candidate_log_shift=self.minlstm_candidate_log_shift,
            ).to(device=target_device, dtype=target_dtype)
        else:
            self.minlstm_pool = MinLSTMPooling(
                input_channels=input_channels,
                hidden_size=self.minlstm_hidden_size,
                forget_bias_init=self.minlstm_forget_bias_init,
                candidate_log_shift=self.minlstm_candidate_log_shift,
            ).to(device=target_device, dtype=target_dtype)

    def _encode(self, x, return_frame_features=False, return_intermediate_features=False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        if return_intermediate_features:
            return {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "frame_map": x,
            }
        if return_frame_features:
            return x  # (B, 1536, T)
        return self.pool_from_frame_features(x)

    def pool_from_frame_features(self, frame_x):
        """Apply configured temporal pooling to frame map (B, 1536, T)."""
        trace = _consume_trace_budget()
        if self.pooling_mode == "attentive_stats":
            t = frame_x.size()[-1]
            global_x = torch.cat(
                (
                    frame_x,
                    torch.mean(frame_x, dim=2, keepdim=True).repeat(1, 1, t),
                    torch.sqrt(torch.var(frame_x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
                ),
                dim=1,
            )
            w = self.attention(global_x)
            mu = torch.sum(frame_x * w, dim=2)
            sg = torch.sqrt((torch.sum((frame_x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
            out = torch.cat((mu, sg), 1)
        else:
            minlstm_input = frame_x
            if self.pooling_mode in {"minlstm_globalproj", "hier_minlstm_globalproj"}:
                t = frame_x.size(-1)
                global_x = torch.cat(
                    (
                        frame_x,
                        torch.mean(frame_x, dim=2, keepdim=True).repeat(1, 1, t),
                        torch.sqrt(torch.var(frame_x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
                    ),
                    dim=1,
                )
                minlstm_input = self.minlstm_global_proj(global_x)
                if trace:
                    print(
                        f"[TRACE ecapa] globalproj {tuple(global_x.shape)} --conv1x1--> {tuple(minlstm_input.shape)}",
                        flush=True,
                    )
            if self.minlstm_pool is None:
                from ..min_lstm.minlstm import (
                    HierarchicalMinLSTMPooling,
                    MinLSTMAttentiveStatsPooling,
                    MinLSTMPooling,
                )

                if self.pooling_mode in {"hier_minlstm", "hier_minlstm_globalproj"}:
                    self.minlstm_pool = HierarchicalMinLSTMPooling(
                        input_channels=minlstm_input.size(1),
                        hidden_size=self.minlstm_hidden_size,
                        forget_bias_init=self.minlstm_forget_bias_init,
                        candidate_log_shift=self.minlstm_candidate_log_shift,
                    ).to(device=frame_x.device, dtype=frame_x.dtype)
                elif self.pooling_mode == "minlstm_asp":
                    self.minlstm_pool = MinLSTMAttentiveStatsPooling(
                        input_channels=minlstm_input.size(1),
                        hidden_size=self.minlstm_hidden_size,
                        forget_bias_init=self.minlstm_forget_bias_init,
                        candidate_log_shift=self.minlstm_candidate_log_shift,
                    ).to(device=frame_x.device, dtype=frame_x.dtype)
                else:
                    self.minlstm_pool = MinLSTMPooling(
                        input_channels=minlstm_input.size(1),
                        hidden_size=self.minlstm_hidden_size,
                        forget_bias_init=self.minlstm_forget_bias_init,
                        candidate_log_shift=self.minlstm_candidate_log_shift,
                    ).to(device=frame_x.device, dtype=frame_x.dtype)
            if trace:
                print(f"[TRACE ecapa] concat_frames{tuple(minlstm_input.shape)} --minlstm_pool--> ...", flush=True)
            out = self.minlstm_pool(minlstm_input)
            if trace:
                print(f"[TRACE ecapa] minlstm_pool output (pre-bn5): {tuple(out.shape)}", flush=True)

        out = self.bn5(out)
        if trace:
            print(f"[TRACE ecapa] after bn5: {tuple(out.shape)}", flush=True)
        out = self.fc6(out)
        if trace:
            print(f"[TRACE ecapa] after fc6: {tuple(out.shape)}", flush=True)
        out = self.bn6(out)
        if trace:
            print(f"[TRACE ecapa] after bn6 (final ECAPA emb): {tuple(out.shape)}", flush=True)
        return out

    def forward(self, x, aug, return_frame_features=False, return_intermediate_features=False):
        """
        Forward pass.

        Args:
            x: If use_melspec=True, raw waves shaped (B, T). Otherwise,
               expects features shaped (B, input_dim, T).
            aug: Whether to apply SpecAugment (only effective when use_melspec=True).
            return_frame_features: If True, return frame-level features (B, 1536, T)
               before attentive statistics pooling.
        """
        if self.use_melspec:
            with torch.no_grad():
                x = self.torchfbank(x) + 1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if aug:
                    x = self.specaug(x)
        else:
            if x.dim() != 3:
                raise ValueError(
                    f"Expected 3D features (B, {self.input_dim}, T) when use_melspec=False, got shape {x.shape}."
                )
        return self._encode(
            x,
            return_frame_features=return_frame_features,
            return_intermediate_features=return_intermediate_features,
        )
