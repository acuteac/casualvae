
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# ACLF 部分多模态锚定因果 VAE
# ------------------------------------------------------------
# 本版本针对用户的真实数据结构设计：
#   - anchors.csv        : 上游临床锚定变量 U
#   - phenotype.csv      : 代理表型 / 结局变量块 Y
#   - protein.csv        : 筛选后的蛋白质，NaN 表示未观测
#   - metabolite.csv     : 筛选后的代谢物，NaN 表示未观测
#
# 相较于原始原型的主要改动：
# 1) 通过基于 NaN 的特征级掩码实现部分多模态支持。
# 2) 临床数据拆分为锚定变量 U 和表型/结局变量 Y。
# 3) 蛋白质/代谢物编码器同时接收数值和掩码。
# 4) 每种组学模态分解为共享因果潜变量 z 和
#    私有干扰潜变量 s。
# 5) SCM 仅在共享的蛋白质/代谢物潜变量上学习。
# 6) 训练分阶段进行：
#      阶段 1：在蛋白质可用行上预训练蛋白质自编码器
#      阶段 2：在代谢物可用行上预训练代谢物自编码器
#      阶段 3：在配对行上进行联合 SCM 训练
# 7) 重构采用掩码机制，损失仅在已观测值上计算。
# 8) 混合表型头同时支持连续型和二分类结局变量。
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_numeric_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "eid" not in df.columns:
        raise ValueError(f"{path} must contain an 'eid' column.")
    return df


def artifact_feature_cols(cols: List[str]) -> List[str]:
    return [c for c in cols if c == "index" or c.startswith("Unnamed")]


def median_fill_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for c in out.columns:
        if out[c].isna().any():
            med = out[c].median()
            if pd.isna(med):
                med = 0.0
            out[c] = out[c].fillna(med)
    return out


def zero_fill_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.fillna(0.0)


def split_indices(indices: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return indices, indices
    if len(indices) == 1 or val_fraction <= 0:
        return indices, np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    n_val = int(round(len(indices) * val_fraction))
    n_val = max(1, min(n_val, len(indices) - 1))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


class PartialMultiOmicsDataset(Dataset):
    def __init__(
        self,
        merged: pd.DataFrame,
        anchor_cols: List[str],
        phenotype_cont_cols: List[str],
        phenotype_bin_cols: List[str],
        protein_cols: List[str],
        metabolite_cols: List[str],
        protein_availability_threshold: float = 0.5,
        metabolite_availability_threshold: float = 0.5,
    ):
        super().__init__()
        self.eids = merged["eid"].to_numpy().astype(np.int64)

        anchors = median_fill_df(merged[anchor_cols]) if anchor_cols else pd.DataFrame(index=merged.index)
        y_cont = median_fill_df(merged[phenotype_cont_cols]) if phenotype_cont_cols else pd.DataFrame(index=merged.index)
        y_bin = zero_fill_df(merged[phenotype_bin_cols]) if phenotype_bin_cols else pd.DataFrame(index=merged.index)

        x_p = merged[protein_cols].copy() if protein_cols else pd.DataFrame(index=merged.index)
        x_m = merged[metabolite_cols].copy() if metabolite_cols else pd.DataFrame(index=merged.index)

        self.anchor_cols = anchor_cols
        self.phenotype_cont_cols = phenotype_cont_cols
        self.phenotype_bin_cols = phenotype_bin_cols
        self.protein_cols = protein_cols
        self.metabolite_cols = metabolite_cols

        self.u = anchors.to_numpy(dtype=np.float32) if anchor_cols else np.zeros((len(merged), 0), dtype=np.float32)
        self.y_cont = y_cont.to_numpy(dtype=np.float32) if phenotype_cont_cols else np.zeros((len(merged), 0), dtype=np.float32)
        self.y_bin = y_bin.to_numpy(dtype=np.float32) if phenotype_bin_cols else np.zeros((len(merged), 0), dtype=np.float32)

        x_p_np = x_p.to_numpy(dtype=np.float32) if protein_cols else np.zeros((len(merged), 0), dtype=np.float32)
        x_m_np = x_m.to_numpy(dtype=np.float32) if metabolite_cols else np.zeros((len(merged), 0), dtype=np.float32)

        self.mask_p = np.isfinite(x_p_np).astype(np.float32)
        self.mask_m = np.isfinite(x_m_np).astype(np.float32)
        self.x_p = np.nan_to_num(x_p_np, nan=0.0)
        self.x_m = np.nan_to_num(x_m_np, nan=0.0)

        frac_p = self.mask_p.mean(axis=1) if self.mask_p.shape[1] > 0 else np.zeros(len(merged), dtype=np.float32)
        frac_m = self.mask_m.mean(axis=1) if self.mask_m.shape[1] > 0 else np.zeros(len(merged), dtype=np.float32)

        self.avail_p = (frac_p >= protein_availability_threshold).astype(np.float32)[:, None]
        self.avail_m = (frac_m >= metabolite_availability_threshold).astype(np.float32)[:, None]
        self.paired = ((self.avail_p[:, 0] > 0.5) & (self.avail_m[:, 0] > 0.5)).astype(np.float32)[:, None]

        self.frac_p = frac_p[:, None].astype(np.float32)
        self.frac_m = frac_m[:, None].astype(np.float32)

    def __len__(self) -> int:
        return len(self.eids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "eid": torch.tensor(self.eids[idx], dtype=torch.long),
            "u": torch.tensor(self.u[idx], dtype=torch.float32),
            "y_cont": torch.tensor(self.y_cont[idx], dtype=torch.float32),
            "y_bin": torch.tensor(self.y_bin[idx], dtype=torch.float32),
            "x_p": torch.tensor(self.x_p[idx], dtype=torch.float32),
            "m_p": torch.tensor(self.mask_p[idx], dtype=torch.float32),
            "x_m": torch.tensor(self.x_m[idx], dtype=torch.float32),
            "m_m": torch.tensor(self.mask_m[idx], dtype=torch.float32),
            "avail_p": torch.tensor(self.avail_p[idx], dtype=torch.float32),
            "avail_m": torch.tensor(self.avail_m[idx], dtype=torch.float32),
            "paired": torch.tensor(self.paired[idx], dtype=torch.float32),
            "frac_p": torch.tensor(self.frac_p[idx], dtype=torch.float32),
            "frac_m": torch.tensor(self.frac_m[idx], dtype=torch.float32),
        }


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedPrivateEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],
        shared_dim: int,
        private_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = MLP(feature_dim * 2, hidden_dims, dropout=dropout)
        hdim = self.backbone.output_dim

        self.shared_mu = nn.Linear(hdim, shared_dim)
        self.shared_logvar = nn.Linear(hdim, shared_dim)
        self.private_mu = nn.Linear(hdim, private_dim)
        self.private_logvar = nn.Linear(hdim, private_dim)

    def forward(self, x_filled: torch.Tensor, x_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(torch.cat([x_filled, x_mask], dim=1))
        return {
            "shared_mu": self.shared_mu(h),
            "shared_logvar": self.shared_logvar(h),
            "private_mu": self.private_mu(h),
            "private_logvar": self.private_logvar(h),
        }


class LinearSharedPrivateDecoder(nn.Module):
    def __init__(self, shared_dim: int, private_dim: int, output_dim: int):
        super().__init__()
        self.shared_linear = nn.Linear(shared_dim, output_dim, bias=False)
        self.private_linear = nn.Linear(private_dim, output_dim, bias=True)

    def forward(self, z_shared: torch.Tensor, z_private: torch.Tensor) -> torch.Tensor:
        return self.shared_linear(z_shared) + self.private_linear(z_private)

    @property
    def shared_weight(self) -> torch.Tensor:
        return self.shared_linear.weight


class MixedOutcomeHead(nn.Module):
    def __init__(self, input_dim: int, n_cont: int, n_bin: int, hidden_dims: List[int]):
        super().__init__()
        self.n_cont = n_cont
        self.n_bin = n_bin
        total_out = n_cont + n_bin
        dims = [input_dim] + hidden_dims + [total_out]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(0.1)])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        cont = out[:, : self.n_cont] if self.n_cont > 0 else out[:, :0]
        binary = out[:, self.n_cont :] if self.n_bin > 0 else out[:, :0]
        return cont, binary


class OutcomeEncoder(nn.Module):
    """
    结局变量编码器（仅共享潜变量，无私有潜变量）
    将连续型和二分类结局编码为共享潜变量
    """
    def __init__(self, n_cont: int, n_bin: int, hidden_dims: List[int], shared_dim: int, dropout: float = 0.1):
        super().__init__()
        input_dim = n_cont + n_bin
        self.backbone = MLP(input_dim, hidden_dims, dropout=dropout)
        hdim = self.backbone.output_dim

        self.shared_mu = nn.Linear(hdim, shared_dim)
        self.shared_logvar = nn.Linear(hdim, shared_dim)

    def forward(self, y_cont: torch.Tensor, y_bin: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_input = torch.cat([y_cont, y_bin], dim=1) if y_cont.numel() > 0 and y_bin.numel() > 0 else (y_cont if y_cont.numel() > 0 else y_bin)
        h = self.backbone(y_input)
        return {
            "shared_mu": self.shared_mu(h),
            "shared_logvar": self.shared_logvar(h),
        }


class OutcomeDecoder(nn.Module):
    """
    结局变量解码器
    从结局共享潜变量重构连续型和二分类结局
    """
    def __init__(self, shared_dim: int, n_cont: int, n_bin: int):
        super().__init__()
        self.n_cont = n_cont
        self.n_bin = n_bin
        total_out = n_cont + n_bin

        # 简单线性投影
        self.linear = nn.Linear(shared_dim, total_out)

    def forward(self, z_outcome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.linear(z_outcome)
        cont = out[:, : self.n_cont] if self.n_cont > 0 else out[:, :0]
        binary = out[:, self.n_cont :] if self.n_bin > 0 else out[:, :0]
        return cont, binary

    @property
    def weight(self) -> torch.Tensor:
        """返回解码器权重矩阵（用于因果图投影）"""
        return self.linear.weight


class BlockSCM(nn.Module):
    """
    共享潜变量结构因果模型 (SCM)：
        z = (I - A^T)^(-1) (eps + context)

    全局图 A 包含 6 个子块（如果包含 outcome）：
        蛋白质->蛋白质
        代谢物->代谢物
        蛋白质->代谢物
        代谢物->蛋白质
        蛋白质->结局
        代谢物->结局

    DAG 惩罚仅在蛋白质和代谢物各自的子块内施加。
    结局节点作为纯下游节点，不允许从结局指向其他节点。
    """

    def __init__(
        self,
        latent_dims: Dict[str, int],
        cross_logit_init: float = -1.8,
        within_logit_init: float = -2.6,
        weight_scale: float = 0.02,
    ):
        super().__init__()
        self.latent_dims = latent_dims
        self.total_dim = sum(latent_dims.values())
        self.block_slices = self._build_block_slices(latent_dims)
        self.has_outcome = "outcome" in latent_dims

        # 基础掩码：对角线为 0
        mask = np.ones((self.total_dim, self.total_dim), dtype=np.float32)
        np.fill_diagonal(mask, 0.0)

        # 如果包含 outcome，强制 outcome -> protein/metabolite 的边为 0
        if self.has_outcome:
            o = self.block_slices["outcome"]
            p = self.block_slices["protein"]
            m = self.block_slices["metabolite"]
            # outcome 不能指向 protein 或 metabolite（行索引为 p/m，列索引为 o）
            mask[p, o] = 0.0
            mask[m, o] = 0.0

        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.register_buffer("eye", torch.eye(self.total_dim, dtype=torch.float32))

        logits = torch.full((self.total_dim, self.total_dim), within_logit_init, dtype=torch.float32)
        p = self.block_slices["protein"]
        m = self.block_slices["metabolite"]
        logits[p, m] = cross_logit_init
        logits[m, p] = cross_logit_init

        # 如果包含 outcome，设置 protein/metabolite -> outcome 的初始 logits
        if self.has_outcome:
            o = self.block_slices["outcome"]
            logits[o, p] = cross_logit_init  # protein -> outcome
            logits[o, m] = cross_logit_init  # metabolite -> outcome

        logits.fill_diagonal_(-12.0)

        self.logits = nn.Parameter(logits)
        self.weights = nn.Parameter(weight_scale * torch.randn(self.total_dim, self.total_dim))

    @staticmethod
    def _build_block_slices(latent_dims: Dict[str, int]) -> Dict[str, slice]:
        slices: Dict[str, slice] = {}
        start = 0
        for name, dim in latent_dims.items():
            slices[name] = slice(start, start + dim)
            start += dim
        return slices

    @staticmethod
    def _sample_logistic_noise_like(x: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(x).clamp(1e-6, 1 - 1e-6)
        return torch.log(u) - torch.log1p(-u)

    def adjacency(self, tau: float = 1.0, hard: bool = False, sample_gumbel: bool = True) -> torch.Tensor:
        logits = self.logits
        if self.training and sample_gumbel:
            logits = logits + self._sample_logistic_noise_like(logits)
        gate = torch.sigmoid(logits / tau)
        if hard:
            gate_hard = (gate > 0.5).float()
            gate = gate_hard.detach() - gate.detach() + gate
        a = gate * self.weights * self.mask
        a = a * (1.0 - torch.eye(self.total_dim, device=a.device))
        return a

    def forward(
        self,
        eps: torch.Tensor,
        context: Optional[torch.Tensor],
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.adjacency(tau=tau, hard=False, sample_gumbel=True)
        rhs = eps if context is None else eps + context
        mat = self.eye.to(rhs.device) - a.T
        z = torch.linalg.solve(mat, rhs.T).T
        return z, a

    def dag_penalty_within_blocks(self, tau: float) -> torch.Tensor:
        a = self.adjacency(tau=tau, hard=False, sample_gumbel=False)
        penalties = []
        for name in ["protein", "metabolite"]:
            sl = self.block_slices[name]
            block = a[sl, sl]
            dim = block.shape[0]
            penalties.append(torch.trace(torch.matrix_exp(block * block)) - dim)
        return torch.stack(penalties).sum()

    def sparsity_penalty(self, tau: float, block_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        a = self.adjacency(tau=tau, hard=False, sample_gumbel=False)
        if not block_weights:
            return torch.abs(a).sum()

        total = torch.tensor(0.0, device=a.device)
        sl = self.block_slices
        block_map = {
            "protein_to_protein": (sl["protein"], sl["protein"]),
            "protein_to_metabolite": (sl["metabolite"], sl["protein"]),   # 目标行, 源列
            "metabolite_to_protein": (sl["protein"], sl["metabolite"]),
            "metabolite_to_metabolite": (sl["metabolite"], sl["metabolite"]),
        }
        if self.has_outcome:
            block_map["protein_to_outcome"] = (sl["outcome"], sl["protein"])
            block_map["metabolite_to_outcome"] = (sl["outcome"], sl["metabolite"])
            block_map["outcome_to_outcome"] = (sl["outcome"], sl["outcome"])

        for key, (row_sl, col_sl) in block_map.items():
            coeff = float(block_weights.get(key, 1.0))
            total = total + coeff * torch.abs(a[row_sl, col_sl]).sum()
        return total

    def spectral_radius_penalty(self, tau: float, margin: float = 0.95) -> torch.Tensor:
        a = self.adjacency(tau=tau, hard=False, sample_gumbel=False)
        eigvals = torch.linalg.eigvals(a)
        rho = torch.max(torch.abs(eigvals)).real
        return F.relu(rho - margin)

    def block_gate_stats(self, tau: float) -> Dict[str, float]:
        with torch.no_grad():
            logits = self.logits
            gate = torch.sigmoid(logits / tau)
            sl = self.block_slices
            stats = {
                "protein_to_protein_gate_mean": float(gate[sl["protein"], sl["protein"]].mean().item()),
                "protein_to_metabolite_gate_mean": float(gate[sl["metabolite"], sl["protein"]].mean().item()),
                "metabolite_to_protein_gate_mean": float(gate[sl["protein"], sl["metabolite"]].mean().item()),
                "metabolite_to_metabolite_gate_mean": float(gate[sl["metabolite"], sl["metabolite"]].mean().item()),
            }
            if self.has_outcome:
                stats["protein_to_outcome_gate_mean"] = float(gate[sl["outcome"], sl["protein"]].mean().item())
                stats["metabolite_to_outcome_gate_mean"] = float(gate[sl["outcome"], sl["metabolite"]].mean().item())
                stats["outcome_to_outcome_gate_mean"] = float(gate[sl["outcome"], sl["outcome"]].mean().item())
        return stats


class PartialAnchoredCausalVAE(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        shared_dims: Dict[str, int],
        private_dims: Dict[str, int],
        hidden_dims: Dict[str, List[int]],
        outcome_hidden_dims: List[int],
    ):
        super().__init__()
        self.input_dims = input_dims
        self.shared_dims = shared_dims
        self.private_dims = private_dims

        self.enc_p = SharedPrivateEncoder(
            feature_dim=input_dims["protein"],
            hidden_dims=hidden_dims["protein"],
            shared_dim=shared_dims["protein"],
            private_dim=private_dims["protein"],
        )
        self.enc_m = SharedPrivateEncoder(
            feature_dim=input_dims["metabolite"],
            hidden_dims=hidden_dims["metabolite"],
            shared_dim=shared_dims["metabolite"],
            private_dim=private_dims["metabolite"],
        )

        # 添加结局编码器（仅共享潜变量）
        self.enc_outcome = OutcomeEncoder(
            n_cont=input_dims["y_cont"],
            n_bin=input_dims["y_bin"],
            hidden_dims=hidden_dims.get("outcome", [64, 32]),
            shared_dim=shared_dims.get("outcome", shared_dims["protein"]),  # 默认与 protein 相同
        )

        self.scm = BlockSCM(latent_dims=shared_dims)

        total_shared = sum(shared_dims.values())
        self.anchor_to_context = nn.Linear(input_dims["anchor"], total_shared, bias=False) if input_dims["anchor"] > 0 else None

        self.dec_p = LinearSharedPrivateDecoder(shared_dims["protein"], private_dims["protein"], input_dims["protein"])
        self.dec_m = LinearSharedPrivateDecoder(shared_dims["metabolite"], private_dims["metabolite"], input_dims["metabolite"])

        # 添加结局解码器
        self.dec_outcome = OutcomeDecoder(
            shared_dim=shared_dims.get("outcome", shared_dims["protein"]),
            n_cont=input_dims["y_cont"],
            n_bin=input_dims["y_bin"],
        )

        # 保留原有的 outcome_head 作为辅助监督（可选）
        outcome_input_dim = total_shared + input_dims["anchor"] + 4
        self.outcome_head = MixedOutcomeHead(
            input_dim=outcome_input_dim,
            n_cont=input_dims["y_cont"],
            n_bin=input_dims["y_bin"],
            hidden_dims=outcome_hidden_dims,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_modality(
        self,
        encoder: SharedPrivateEncoder,
        x_filled: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = encoder(x_filled, x_mask)
        shared_eps = self.reparameterize(out["shared_mu"], out["shared_logvar"])
        private = self.reparameterize(out["private_mu"], out["private_logvar"])
        out["shared_eps"] = shared_eps
        out["private"] = private
        return out

    def encode_protein(self, x_p: torch.Tensor, m_p: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encode_modality(self.enc_p, x_p, m_p)

    def encode_metabolite(self, x_m: torch.Tensor, m_m: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encode_modality(self.enc_m, x_m, m_m)

    def encode_outcome(self, y_cont: torch.Tensor, y_bin: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码结局变量（仅共享潜变量）"""
        out = self.enc_outcome(y_cont, y_bin)
        shared_eps = self.reparameterize(out["shared_mu"], out["shared_logvar"])
        out["shared_eps"] = shared_eps
        return out

    def get_causal_decoder_projection(self) -> torch.Tensor:
        """获取因果解码器投影矩阵（包含结局）"""
        if "outcome" in self.shared_dims:
            return torch.block_diag(
                self.dec_p.shared_weight,
                self.dec_m.shared_weight,
                self.dec_outcome.weight
            )
        else:
            return torch.block_diag(self.dec_p.shared_weight, self.dec_m.shared_weight)

    def forward_protein_only(self, x_p: torch.Tensor, m_p: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_p = self.encode_protein(x_p, m_p)
        xhat_p = self.dec_p(enc_p["shared_eps"], enc_p["private"])
        return {"enc_p": enc_p, "xhat_p": xhat_p}

    def forward_metabolite_only(self, x_m: torch.Tensor, m_m: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_m = self.encode_metabolite(x_m, m_m)
        xhat_m = self.dec_m(enc_m["shared_eps"], enc_m["private"])
        return {"enc_m": enc_m, "xhat_m": xhat_m}

    def forward_joint(
        self,
        u: torch.Tensor,
        y_cont: torch.Tensor,
        y_bin: torch.Tensor,
        x_p: torch.Tensor,
        m_p: torch.Tensor,
        x_m: torch.Tensor,
        m_m: torch.Tensor,
        avail_p: torch.Tensor,
        avail_m: torch.Tensor,
        frac_p: torch.Tensor,
        frac_m: torch.Tensor,
        use_scm: bool,
        tau: float,
    ) -> Dict[str, torch.Tensor]:
        enc_p = self.encode_protein(x_p, m_p)
        enc_m = self.encode_metabolite(x_m, m_m)
        enc_outcome = self.encode_outcome(y_cont, y_bin)

        # 拼接所有模态的共享潜变量
        eps_parts = [enc_p["shared_eps"] * avail_p, enc_m["shared_eps"] * avail_m]
        if "outcome" in self.shared_dims:
            eps_parts.append(enc_outcome["shared_eps"])
        eps_shared = torch.cat(eps_parts, dim=1)

        context = self.anchor_to_context(u) if self.anchor_to_context is not None else None

        if use_scm:
            z_all, a = self.scm(eps_shared, context=context, tau=tau)
        else:
            z_all = eps_shared if context is None else eps_shared + context
            a = self.scm.adjacency(tau=tau, hard=False, sample_gumbel=False)

        dp = self.shared_dims["protein"]
        dm = self.shared_dims["metabolite"]
        z_p = z_all[:, :dp] * avail_p
        z_m = z_all[:, dp : dp + dm] * avail_m

        z_outcome = None
        if "outcome" in self.shared_dims:
            do = self.shared_dims["outcome"]
            z_outcome = z_all[:, dp + dm : dp + dm + do]

        xhat_p = self.dec_p(z_p, enc_p["private"])
        xhat_m = self.dec_m(z_m, enc_m["private"])

        # 优先使用因果图中的 outcome 潜变量解码
        if z_outcome is not None:
            y_cont_hat, y_bin_logits = self.dec_outcome(z_outcome)
        else:
            outcome_input = torch.cat([z_p, z_m, u, avail_p, avail_m, frac_p, frac_m], dim=1)
            y_cont_hat, y_bin_logits = self.outcome_head(outcome_input)

        return {
            "enc_p": enc_p,
            "enc_m": enc_m,
            "enc_outcome": enc_outcome,
            "z_all": z_all,
            "z_p": z_p,
            "z_m": z_m,
            "z_outcome": z_outcome,
            "a": a,
            "xhat_p": xhat_p,
            "xhat_m": xhat_m,
            "y_cont_hat": y_cont_hat,
            "y_bin_logits": y_bin_logits,
        }


@dataclass
class TrainConfig:
    batch_size: int = 32
    val_fraction: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    pretrain_protein_epochs: int = 35
    pretrain_metabolite_epochs: int = 35
    joint_epochs: int = 180
    joint_warmup_epochs: int = 20
    ramp_epochs: int = 60

    lr_pretrain: float = 1e-3
    lr_joint: float = 7e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    tau_start: float = 2.0
    tau_end: float = 0.5

    lambda_kl_shared: float = 1e-3
    lambda_kl_private: float = 1e-3
    lambda_ind: float = 5e-3
    lambda_sparse: float = 4e-4
    lambda_dag: float = 5e-3
    lambda_stable: float = 1.0
    lambda_ortho: float = 1e-2
    lambda_decoder_sparse: float = 5e-5
    lambda_outcome_cont: float = 1.0
    lambda_outcome_bin: float = 0.3

    early_stopping_patience: int = 25


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def orthogonality_penalty(weight: torch.Tensor) -> torch.Tensor:
    gram = weight.T @ weight
    ident = torch.eye(gram.shape[0], device=weight.device)
    return ((gram - ident) ** 2).mean()


def covariance_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(0.0, device=x.device if x.numel() else y.device)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov = (x.T @ y) / max(x.shape[0] - 1, 1)
    return (cov ** 2).mean()


def cosine_tau(step: int, total_steps: int, tau_start: float, tau_end: float) -> float:
    if total_steps <= 1:
        return tau_end
    ratio = step / float(total_steps - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * ratio))
    return tau_end + (tau_start - tau_end) * cosine


def ramp_value(epoch: int, start_epoch: int, ramp_epochs: int, max_value: float) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return max_value
    alpha = min((epoch - start_epoch + 1) / float(ramp_epochs), 1.0)
    return alpha * max_value


def decoder_sparsity_penalty(model: PartialAnchoredCausalVAE) -> torch.Tensor:
    return torch.abs(model.dec_p.shared_weight).sum() + torch.abs(model.dec_m.shared_weight).sum()


def make_dataloaders(dataset: PartialMultiOmicsDataset, cfg: TrainConfig, seed: int):
    idx_all = np.arange(len(dataset))
    idx_protein = idx_all[dataset.avail_p[:, 0] > 0.5]
    idx_metabolite = idx_all[dataset.avail_m[:, 0] > 0.5]
    idx_paired = idx_all[dataset.paired[:, 0] > 0.5]

    p_train, p_val = split_indices(idx_protein, cfg.val_fraction, seed)
    m_train, m_val = split_indices(idx_metabolite, cfg.val_fraction, seed + 1)
    pair_train, pair_val = split_indices(idx_paired, cfg.val_fraction, seed + 2)

    loaders = {
        "protein_train": DataLoader(Subset(dataset, p_train.tolist()), batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        "protein_val": DataLoader(Subset(dataset, p_val.tolist()), batch_size=cfg.batch_size, shuffle=False, drop_last=True),
        "metabolite_train": DataLoader(Subset(dataset, m_train.tolist()), batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        "metabolite_val": DataLoader(Subset(dataset, m_val.tolist()), batch_size=cfg.batch_size, shuffle=False, drop_last=True),
        "paired_train": DataLoader(Subset(dataset, pair_train.tolist()), batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        "paired_val": DataLoader(Subset(dataset, pair_val.tolist()), batch_size=cfg.batch_size, shuffle=False, drop_last=True),
        "all": DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False),
    }
    split_meta = {
        "n_total": int(len(dataset)),
        "n_protein_available": int(len(idx_protein)),
        "n_metabolite_available": int(len(idx_metabolite)),
        "n_paired_available": int(len(idx_paired)),
        "n_protein_train": int(len(p_train)),
        "n_protein_val": int(len(p_val)),
        "n_metabolite_train": int(len(m_train)),
        "n_metabolite_val": int(len(m_val)),
        "n_paired_train": int(len(pair_train)),
        "n_paired_val": int(len(pair_val)),
    }
    return loaders, split_meta


def modality_pretrain_loss(
    model: PartialAnchoredCausalVAE,
    batch: Dict[str, torch.Tensor],
    modality: str,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if modality == "protein":
        x = batch["x_p"].to(cfg.device)
        m = batch["m_p"].to(cfg.device)
        out = model.forward_protein_only(x, m)
        recon = masked_mse(out["xhat_p"], x, m)
        kl_shared = kl_normal(out["enc_p"]["shared_mu"], out["enc_p"]["shared_logvar"])
        kl_private = kl_normal(out["enc_p"]["private_mu"], out["enc_p"]["private_logvar"])
        ind = covariance_penalty(out["enc_p"]["shared_eps"], out["enc_p"]["private"])
        ortho = orthogonality_penalty(model.dec_p.shared_weight)
    else:
        x = batch["x_m"].to(cfg.device)
        m = batch["m_m"].to(cfg.device)
        out = model.forward_metabolite_only(x, m)
        recon = masked_mse(out["xhat_m"], x, m)
        kl_shared = kl_normal(out["enc_m"]["shared_mu"], out["enc_m"]["shared_logvar"])
        kl_private = kl_normal(out["enc_m"]["private_mu"], out["enc_m"]["private_logvar"])
        ind = covariance_penalty(out["enc_m"]["shared_eps"], out["enc_m"]["private"])
        ortho = orthogonality_penalty(model.dec_m.shared_weight)

    loss = (
        recon
        + cfg.lambda_kl_shared * kl_shared
        + cfg.lambda_kl_private * kl_private
        + cfg.lambda_ind * ind
        + cfg.lambda_ortho * ortho
        + cfg.lambda_decoder_sparse * decoder_sparsity_penalty(model)
    )
    scalars = {
        "loss": float(loss.item()),
        "recon": float(recon.item()),
        "kl_shared": float(kl_shared.item()),
        "kl_private": float(kl_private.item()),
        "ind": float(ind.item()),
        "ortho": float(ortho.item()),
    }
    return loss, scalars


def joint_loss(
    model: PartialAnchoredCausalVAE,
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    epoch: int,
    block_sparsity_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    u = batch["u"].to(cfg.device)
    y_cont = batch["y_cont"].to(cfg.device)
    y_bin = batch["y_bin"].to(cfg.device)

    x_p = batch["x_p"].to(cfg.device)
    m_p = batch["m_p"].to(cfg.device)
    x_m = batch["x_m"].to(cfg.device)
    m_m = batch["m_m"].to(cfg.device)

    avail_p = batch["avail_p"].to(cfg.device)
    avail_m = batch["avail_m"].to(cfg.device)
    frac_p = batch["frac_p"].to(cfg.device)
    frac_m = batch["frac_m"].to(cfg.device)

    tau = cosine_tau(epoch, cfg.joint_epochs, cfg.tau_start, cfg.tau_end)
    use_scm = epoch >= cfg.joint_warmup_epochs

    out = model.forward_joint(
        u=u,
        y_cont=y_cont,
        y_bin=y_bin,
        x_p=x_p,
        m_p=m_p,
        x_m=x_m,
        m_m=m_m,
        avail_p=avail_p,
        avail_m=avail_m,
        frac_p=frac_p,
        frac_m=frac_m,
        use_scm=use_scm,
        tau=tau,
    )

    recon_p = masked_mse(out["xhat_p"], x_p, m_p)
    recon_m = masked_mse(out["xhat_m"], x_m, m_m)
    recon = recon_p + recon_m

    kl_shared = kl_normal(out["enc_p"]["shared_mu"], out["enc_p"]["shared_logvar"]) + kl_normal(
        out["enc_m"]["shared_mu"], out["enc_m"]["shared_logvar"]
    )
    # 添加结局 KL 散度
    if "enc_outcome" in out and out["enc_outcome"] is not None:
        kl_shared = kl_shared + kl_normal(out["enc_outcome"]["shared_mu"], out["enc_outcome"]["shared_logvar"])

    kl_private = kl_normal(out["enc_p"]["private_mu"], out["enc_p"]["private_logvar"]) + kl_normal(
        out["enc_m"]["private_mu"], out["enc_m"]["private_logvar"]
    )

    ind = (
        covariance_penalty(out["z_p"], out["enc_p"]["private"])
        + covariance_penalty(out["z_m"], out["enc_m"]["private"])
        + covariance_penalty(out["enc_p"]["private"], out["enc_m"]["private"])
    )

    dag = model.scm.dag_penalty_within_blocks(tau=tau) if use_scm else torch.tensor(0.0, device=cfg.device)
    sparse = model.scm.sparsity_penalty(tau=tau, block_weights=block_sparsity_weights) if use_scm else torch.tensor(0.0, device=cfg.device)
    stable = model.scm.spectral_radius_penalty(tau=tau) if use_scm else torch.tensor(0.0, device=cfg.device)

    ortho = orthogonality_penalty(model.dec_p.shared_weight) + orthogonality_penalty(model.dec_m.shared_weight)
    # 添加结局解码器的正交性惩罚
    if "outcome" in model.shared_dims:
        ortho = ortho + orthogonality_penalty(model.dec_outcome.weight)

    dec_sparse = decoder_sparsity_penalty(model)
    # 添加结局解码器的稀疏性惩罚
    if "outcome" in model.shared_dims:
        dec_sparse = dec_sparse + torch.abs(model.dec_outcome.weight).sum()

    cont_loss = F.mse_loss(out["y_cont_hat"], y_cont) if y_cont.numel() > 0 else torch.tensor(0.0, device=cfg.device)
    bin_loss = F.binary_cross_entropy_with_logits(out["y_bin_logits"], y_bin) if y_bin.numel() > 0 else torch.tensor(0.0, device=cfg.device)

    lambda_dag = ramp_value(epoch, cfg.joint_warmup_epochs, cfg.ramp_epochs, cfg.lambda_dag)
    lambda_sparse = ramp_value(epoch, cfg.joint_warmup_epochs, cfg.ramp_epochs, cfg.lambda_sparse)
    lambda_stable = ramp_value(epoch, cfg.joint_warmup_epochs, cfg.ramp_epochs, cfg.lambda_stable)
    lambda_kl_shared = ramp_value(epoch, 0, cfg.ramp_epochs, cfg.lambda_kl_shared)
    lambda_kl_private = ramp_value(epoch, 0, cfg.ramp_epochs, cfg.lambda_kl_private)
    lambda_ind = ramp_value(epoch, cfg.joint_warmup_epochs // 2, cfg.ramp_epochs, cfg.lambda_ind)

    loss = (
        recon
        + lambda_kl_shared * kl_shared
        + lambda_kl_private * kl_private
        + lambda_ind * ind
        + lambda_sparse * sparse
        + lambda_dag * dag
        + lambda_stable * stable
        + cfg.lambda_ortho * ortho
        + cfg.lambda_decoder_sparse * dec_sparse
        + cfg.lambda_outcome_cont * cont_loss
        + cfg.lambda_outcome_bin * bin_loss
    )

    gate_stats = model.scm.block_gate_stats(tau=tau)
    scalars = {
        "loss": float(loss.item()),
        "recon": float(recon.item()),
        "recon_p": float(recon_p.item()),
        "recon_m": float(recon_m.item()),
        "kl_shared": float(kl_shared.item()),
        "kl_private": float(kl_private.item()),
        "ind": float(ind.item()),
        "dag": float(dag.item()),
        "sparse": float(sparse.item()),
        "stable": float(stable.item()),
        "ortho": float(ortho.item()),
        "dec_sparse": float(dec_sparse.item()),
        "outcome_cont": float(cont_loss.item()),
        "outcome_bin": float(bin_loss.item()),
        "tau": float(tau),
        "use_scm": float(use_scm),
    }
    scalars.update(gate_stats)
    return loss, scalars


def run_pretrain_epoch(
    model: PartialAnchoredCausalVAE,
    loader: DataLoader,
    modality: str,
    cfg: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    # 始终保持 train 模式以使用 BatchNorm 的 running stats，
    # 避免小验证集导致 BN 统计量异常
    model.train(True)

    running = {"loss": 0.0, "recon": 0.0, "kl_shared": 0.0, "kl_private": 0.0, "ind": 0.0, "ortho": 0.0}
    n_batches = 0
    for batch in loader:
        if training:
            optimizer.zero_grad()
        loss, scalars = modality_pretrain_loss(model, batch, modality, cfg)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        for k, v in scalars.items():
            running[k] += v
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in running}
    return {k: v / n_batches for k, v in running.items()}


def run_joint_epoch(
    model: PartialAnchoredCausalVAE,
    loader: DataLoader,
    cfg: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    block_sparsity_weights: Dict[str, float],
) -> Dict[str, float]:
    training = optimizer is not None
    # 始终保持 train 模式以使用 BatchNorm 的 running stats，
    # 避免小验证集导致 BN 统计量异常
    model.train(True)

    running: Dict[str, float] = {
        "loss": 0.0, "recon": 0.0, "recon_p": 0.0, "recon_m": 0.0,
        "kl_shared": 0.0, "kl_private": 0.0, "ind": 0.0,
        "dag": 0.0, "sparse": 0.0, "stable": 0.0,
        "ortho": 0.0, "dec_sparse": 0.0,
        "outcome_cont": 0.0, "outcome_bin": 0.0,
        "tau": 0.0, "use_scm": 0.0,
        "protein_to_protein_gate_mean": 0.0,
        "protein_to_metabolite_gate_mean": 0.0,
        "metabolite_to_protein_gate_mean": 0.0,
        "metabolite_to_metabolite_gate_mean": 0.0,
    }
    # 如果模型包含 outcome，添加对应的 gate stats 键
    if "outcome" in model.shared_dims:
        running["protein_to_outcome_gate_mean"] = 0.0
        running["metabolite_to_outcome_gate_mean"] = 0.0
        running["outcome_to_outcome_gate_mean"] = 0.0

    n_batches = 0

    for batch in loader:
        if training:
            optimizer.zero_grad()
        loss, scalars = joint_loss(model, batch, cfg, epoch, block_sparsity_weights)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        for k, v in scalars.items():
            if k in running:
                running[k] += v
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in running}
    return {k: v / n_batches for k, v in running.items()}


def train_model(
    model: PartialAnchoredCausalVAE,
    loaders: Dict[str, DataLoader],
    cfg: TrainConfig,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    model.to(cfg.device)

    # -------- 阶段 1：蛋白质预训练 --------
    opt_p = torch.optim.AdamW(
        list(model.enc_p.parameters()) + list(model.dec_p.parameters()),
        lr=cfg.lr_pretrain,
        weight_decay=cfg.weight_decay,
    )
    for epoch in range(cfg.pretrain_protein_epochs):
        train_metrics = run_pretrain_epoch(model, loaders["protein_train"], "protein", cfg, opt_p)
        with torch.no_grad():
            val_metrics = run_pretrain_epoch(model, loaders["protein_val"], "protein", cfg, None)
        row = {"stage": "pretrain_protein", "epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == cfg.pretrain_protein_epochs - 1:
            print(f"[Protein pretrain] epoch {epoch+1:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}")

    # -------- 阶段 2：代谢物预训练 --------
    opt_m = torch.optim.AdamW(
        list(model.enc_m.parameters()) + list(model.dec_m.parameters()),
        lr=cfg.lr_pretrain,
        weight_decay=cfg.weight_decay,
    )
    for epoch in range(cfg.pretrain_metabolite_epochs):
        train_metrics = run_pretrain_epoch(model, loaders["metabolite_train"], "metabolite", cfg, opt_m)
        with torch.no_grad():
            val_metrics = run_pretrain_epoch(model, loaders["metabolite_val"], "metabolite", cfg, None)
        row = {"stage": "pretrain_metabolite", "epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == cfg.pretrain_metabolite_epochs - 1:
            print(f"[Metabolite pretrain] epoch {epoch+1:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}")

    # -------- 阶段 3：配对联合训练 --------
    opt_joint = torch.optim.AdamW(model.parameters(), lr=cfg.lr_joint, weight_decay=cfg.weight_decay)
    block_sparsity_weights = {
        "protein_to_protein": 1.0,
        "metabolite_to_metabolite": 1.0,
        "protein_to_metabolite": 0.35,
        "metabolite_to_protein": 0.35,
    }
    # 如果模型包含 outcome 节点，添加对应的稀疏性权重
    if "outcome" in model.shared_dims:
        block_sparsity_weights["protein_to_outcome"] = 0.5
        block_sparsity_weights["metabolite_to_outcome"] = 0.5
        block_sparsity_weights["outcome_to_outcome"] = 999.0

    best_state = None
    best_val = float("inf")
    patience = 0

    for epoch in range(cfg.joint_epochs):
        train_metrics = run_joint_epoch(model, loaders["paired_train"], cfg, opt_joint, epoch, block_sparsity_weights)
        with torch.no_grad():
            val_metrics = run_joint_epoch(model, loaders["paired_val"], cfg, None, epoch, block_sparsity_weights)

        row = {"stage": "joint", "epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        # 当验证 loss 为 nan/inf 时回退到训练 loss 进行 early stopping
        current_val = val_metrics["loss"]
        if not (math.isfinite(current_val)):
            current_val = train_metrics["loss"]

        if current_val < best_val:
            best_val = current_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == cfg.joint_epochs - 1:
            print(
                f"[Joint] epoch {epoch+1:03d} "
                f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"recon={train_metrics['recon']:.4f} cont={train_metrics['outcome_cont']:.4f} "
                f"bin={train_metrics['outcome_bin']:.4f} dag={train_metrics['dag']:.4f} "
                f"tau={train_metrics['tau']:.3f} scm={bool(train_metrics['use_scm'])}"
            )

        if patience >= cfg.early_stopping_patience:
            print(f"Early stopping joint training at epoch {epoch+1:03d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def extract_outputs(
    model: PartialAnchoredCausalVAE,
    dataset: PartialMultiOmicsDataset,
    loader: DataLoader,
    out_dir: str,
    tau: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    model.to(next(model.parameters()).device)

    rows = []
    z_list = []
    ycont_hat_list = []
    ybin_hat_list = []

    for batch in loader:
        u = batch["u"].to(next(model.parameters()).device)
        y_cont = batch["y_cont"].to(next(model.parameters()).device)
        y_bin = batch["y_bin"].to(next(model.parameters()).device)
        x_p = batch["x_p"].to(next(model.parameters()).device)
        m_p = batch["m_p"].to(next(model.parameters()).device)
        x_m = batch["x_m"].to(next(model.parameters()).device)
        m_m = batch["m_m"].to(next(model.parameters()).device)
        avail_p = batch["avail_p"].to(next(model.parameters()).device)
        avail_m = batch["avail_m"].to(next(model.parameters()).device)
        frac_p = batch["frac_p"].to(next(model.parameters()).device)
        frac_m = batch["frac_m"].to(next(model.parameters()).device)

        out = model.forward_joint(
            u=u, y_cont=y_cont, y_bin=y_bin,
            x_p=x_p, m_p=m_p, x_m=x_m, m_m=m_m,
            avail_p=avail_p, avail_m=avail_m,
            frac_p=frac_p, frac_m=frac_m,
            use_scm=True, tau=tau
        )
        z_list.append(out["z_all"].cpu().numpy())
        ycont_hat_list.append(out["y_cont_hat"].cpu().numpy())
        ybin_hat_list.append(torch.sigmoid(out["y_bin_logits"]).cpu().numpy())

        rows.append(pd.DataFrame({
            "eid": batch["eid"].cpu().numpy(),
            "protein_available": batch["avail_p"].cpu().numpy().reshape(-1),
            "metabolite_available": batch["avail_m"].cpu().numpy().reshape(-1),
            "paired_available": batch["paired"].cpu().numpy().reshape(-1),
            "protein_observed_fraction": batch["frac_p"].cpu().numpy().reshape(-1),
            "metabolite_observed_fraction": batch["frac_m"].cpu().numpy().reshape(-1),
        }))

    sample_meta = pd.concat(rows, axis=0, ignore_index=True)
    pd.DataFrame(sample_meta).to_csv(os.path.join(out_dir, "sample_availability.csv"), index=False)

    latent = np.concatenate(z_list, axis=0)
    latent_col_names = [f"P{i+1}" for i in range(model.shared_dims["protein"])] + [f"M{i+1}" for i in range(model.shared_dims["metabolite"])]
    if "outcome" in model.shared_dims:
        latent_col_names += [f"O{i+1}" for i in range(model.shared_dims["outcome"])]
    pd.DataFrame(latent, columns=latent_col_names).to_csv(
        os.path.join(out_dir, "shared_latent_z.csv"), index=False
    )

    ycont_hat = np.concatenate(ycont_hat_list, axis=0) if ycont_hat_list else np.zeros((len(dataset), 0), dtype=np.float32)
    ybin_hat = np.concatenate(ybin_hat_list, axis=0) if ybin_hat_list else np.zeros((len(dataset), 0), dtype=np.float32)

    if dataset.phenotype_cont_cols:
        pd.DataFrame(ycont_hat, columns=dataset.phenotype_cont_cols).to_csv(
            os.path.join(out_dir, "phenotype_cont_pred.csv"), index=False
        )
    if dataset.phenotype_bin_cols:
        pd.DataFrame(ybin_hat, columns=dataset.phenotype_bin_cols).to_csv(
            os.path.join(out_dir, "phenotype_bin_pred_prob.csv"), index=False
        )

    a = model.scm.adjacency(tau=tau, hard=False, sample_gumbel=False).detach().cpu().numpy()
    d = model.get_causal_decoder_projection().detach().cpu().numpy()
    g = d @ a @ d.T

    shared_names = [f"P{i+1}" for i in range(model.shared_dims["protein"])] + [f"M{i+1}" for i in range(model.shared_dims["metabolite"])]
    if "outcome" in model.shared_dims:
        shared_names += [f"O{i+1}" for i in range(model.shared_dims["outcome"])]

    outcome_feature_names = dataset.phenotype_cont_cols + dataset.phenotype_bin_cols if "outcome" in model.shared_dims else []
    feature_names = dataset.protein_cols + dataset.metabolite_cols + outcome_feature_names

    pd.DataFrame(a, index=shared_names, columns=shared_names).to_csv(
        os.path.join(out_dir, "latent_causal_A.csv")
    )
    pd.DataFrame(d, index=feature_names, columns=shared_names).to_csv(
        os.path.join(out_dir, "decoder_projection_D.csv")
    )
    pd.DataFrame(g, index=feature_names, columns=feature_names).to_csv(
        os.path.join(out_dir, "feature_projection_G.csv")
    )

    p_feat = dataset.protein_cols
    m_feat = dataset.metabolite_cols
    o_feat = outcome_feature_names
    p_idx = slice(0, len(p_feat))
    m_idx = slice(len(p_feat), len(p_feat) + len(m_feat))
    o_idx = slice(len(p_feat) + len(m_feat), len(p_feat) + len(m_feat) + len(o_feat))

    g_pp = g[p_idx, p_idx]
    g_pm = g[p_idx, m_idx]
    g_mp = g[m_idx, p_idx]
    g_mm = g[m_idx, m_idx]

    pd.DataFrame(g_pp, index=p_feat, columns=p_feat).to_csv(os.path.join(out_dir, "G_protein_to_protein.csv"))
    pd.DataFrame(g_pm, index=p_feat, columns=m_feat).to_csv(os.path.join(out_dir, "G_metabolite_to_protein.csv"))
    pd.DataFrame(g_mp, index=m_feat, columns=p_feat).to_csv(os.path.join(out_dir, "G_protein_to_metabolite.csv"))
    pd.DataFrame(g_mm, index=m_feat, columns=m_feat).to_csv(os.path.join(out_dir, "G_metabolite_to_metabolite.csv"))

    # 导出结局相关的因果边
    if o_feat:
        g_po = g[o_idx, p_idx]  # protein -> outcome
        g_mo = g[o_idx, m_idx]  # metabolite -> outcome
        pd.DataFrame(g_po, index=o_feat, columns=p_feat).to_csv(os.path.join(out_dir, "G_protein_to_outcome.csv"))
        pd.DataFrame(g_mo, index=o_feat, columns=m_feat).to_csv(os.path.join(out_dir, "G_metabolite_to_outcome.csv"))

    def top_edges(block: np.ndarray, row_names: List[str], col_names: List[str], block_name: str, k: int = 50) -> pd.DataFrame:
        # 向量化实现，避免大矩阵嵌套循环导致 MemoryError
        abs_block = np.abs(block)
        same_names = row_names is col_names
        if same_names:
            np.fill_diagonal(abs_block, 0.0)

        # 扁平化后取 top-k 索引
        flat = abs_block.ravel()
        n = min(k, flat.size)
        if n == 0:
            return pd.DataFrame(columns=["rank", "source", "target", "edge_weight", "abs_edge_weight", "block"])
        top_idx = np.argpartition(flat, -n)[-n:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

        row_idx, col_idx = np.unravel_index(top_idx, block.shape)
        df = pd.DataFrame({
            "rank": np.arange(1, n + 1),
            "source": [col_names[j] for j in col_idx],
            "target": [row_names[i] for i in row_idx],
            "edge_weight": block[row_idx, col_idx].astype(float),
            "abs_edge_weight": abs_block[row_idx, col_idx].astype(float),
            "block": block_name,
        })
        return df

    top_edges(g_pp, p_feat, p_feat, "protein_to_protein").to_csv(os.path.join(out_dir, "top_edges_protein_to_protein.csv"), index=False)
    top_edges(g_pm, p_feat, m_feat, "metabolite_to_protein").to_csv(os.path.join(out_dir, "top_edges_metabolite_to_protein.csv"), index=False)
    top_edges(g_mp, m_feat, p_feat, "protein_to_metabolite").to_csv(os.path.join(out_dir, "top_edges_protein_to_metabolite.csv"), index=False)
    top_edges(g_mm, m_feat, m_feat, "metabolite_to_metabolite").to_csv(os.path.join(out_dir, "top_edges_metabolite_to_metabolite.csv"), index=False)

    # 导出结局相关的 top edges
    if o_feat:
        top_edges(g_po, o_feat, p_feat, "protein_to_outcome").to_csv(os.path.join(out_dir, "top_edges_protein_to_outcome.csv"), index=False)
        top_edges(g_mo, o_feat, m_feat, "metabolite_to_outcome").to_csv(os.path.join(out_dir, "top_edges_metabolite_to_outcome.csv"), index=False)

    summary = {
        "n_samples": int(len(dataset)),
        "anchor_features": len(dataset.anchor_cols),
        "phenotype_cont_features": len(dataset.phenotype_cont_cols),
        "phenotype_bin_features": len(dataset.phenotype_bin_cols),
        "protein_features": len(dataset.protein_cols),
        "metabolite_features": len(dataset.metabolite_cols),
        "protein_available_ge_threshold": int((dataset.avail_p[:, 0] > 0.5).sum()),
        "metabolite_available_ge_threshold": int((dataset.avail_m[:, 0] > 0.5).sum()),
        "paired_available_ge_threshold": int((dataset.paired[:, 0] > 0.5).sum()),
        "shared_dims": model.shared_dims,
        "private_dims": model.private_dims,
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACLF partial multimodal anchored causal VAE")
    parser.add_argument("--anchors_csv", type=str, required=True)
    parser.add_argument("--phenotype_csv", type=str, required=True)
    parser.add_argument("--protein_csv", type=str, required=True)
    parser.add_argument("--metabolite_csv", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--shared_p", type=int, default=6)
    parser.add_argument("--shared_m", type=int, default=6)
    parser.add_argument("--private_p", type=int, default=3)
    parser.add_argument("--private_m", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_fraction", type=float, default=0.2)

    parser.add_argument("--pretrain_protein_epochs", type=int, default=35)
    parser.add_argument("--pretrain_metabolite_epochs", type=int, default=35)
    parser.add_argument("--joint_epochs", type=int, default=180)
    parser.add_argument("--joint_warmup_epochs", type=int, default=20)
    parser.add_argument("--ramp_epochs", type=int, default=60)

    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_joint", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--tau_start", type=float, default=2.0)
    parser.add_argument("--tau_end", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_structured_inputs(args):
    config = json.load(open(args.config_json, "r", encoding="utf-8"))

    anchors_df = read_numeric_csv(args.anchors_csv)
    phenotype_df = read_numeric_csv(args.phenotype_csv)
    protein_df = read_numeric_csv(args.protein_csv)
    metabolite_df = read_numeric_csv(args.metabolite_csv)

    bad_feature_cols = artifact_feature_cols(config["protein_cols"]) + artifact_feature_cols(config["metabolite_cols"])
    if bad_feature_cols:
        raise ValueError(f"Artifact columns leaked into feature config: {bad_feature_cols}")

    merged = anchors_df.merge(phenotype_df, on="eid", how="inner") \
                       .merge(protein_df, on="eid", how="left") \
                       .merge(metabolite_df, on="eid", how="left")

    dataset = PartialMultiOmicsDataset(
        merged=merged,
        anchor_cols=config["anchor_cols"],
        phenotype_cont_cols=config["phenotype_continuous_cols"],
        phenotype_bin_cols=config.get("phenotype_binary_cols", []),
        protein_cols=config["protein_cols"],
        metabolite_cols=config["metabolite_cols"],
        protein_availability_threshold=float(config.get("protein_availability_threshold", 0.5)),
        metabolite_availability_threshold=float(config.get("metabolite_availability_threshold", 0.5)),
    )
    return dataset, config


def main():
    parser = build_argparser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    dataset, config = load_structured_inputs(args)
    cfg = TrainConfig(
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        device=args.device,
        pretrain_protein_epochs=args.pretrain_protein_epochs,
        pretrain_metabolite_epochs=args.pretrain_metabolite_epochs,
        joint_epochs=args.joint_epochs,
        joint_warmup_epochs=args.joint_warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        lr_pretrain=args.lr_pretrain,
        lr_joint=args.lr_joint,
        weight_decay=args.weight_decay,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
    )

    loaders, split_meta = make_dataloaders(dataset, cfg, args.seed)

    input_dims = {
        "anchor": len(dataset.anchor_cols),
        "y_cont": len(dataset.phenotype_cont_cols),
        "y_bin": len(dataset.phenotype_bin_cols),
        "protein": len(dataset.protein_cols),
        "metabolite": len(dataset.metabolite_cols),
    }
    shared_dims = {
        "protein": int(config.get("shared_dims", {}).get("protein", args.shared_p)),
        "metabolite": int(config.get("shared_dims", {}).get("metabolite", args.shared_m)),
    }
    # 如果配置中包含 outcome 共享维度，则添加到 shared_dims
    if config.get("shared_dims", {}).get("outcome"):
        shared_dims["outcome"] = int(config["shared_dims"]["outcome"])
    private_dims = {
        "protein": int(config.get("private_dims", {}).get("protein", args.private_p)),
        "metabolite": int(config.get("private_dims", {}).get("metabolite", args.private_m)),
    }
    hidden_dims = config.get(
        "hidden_dims",
        {
            "protein": [128, 64],
            "metabolite": [128, 64],
        },
    )
    outcome_hidden_dims = config.get("outcome_hidden_dims", [64, 32])

    model = PartialAnchoredCausalVAE(
        input_dims=input_dims,
        shared_dims=shared_dims,
        private_dims=private_dims,
        hidden_dims=hidden_dims,
        outcome_hidden_dims=outcome_hidden_dims,
    )

    history = train_model(model, loaders, cfg)
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    extract_outputs(
        model=model,
        dataset=dataset,
        loader=loaders["all"],
        out_dir=args.out_dir,
        tau=cfg.tau_end,
    )

    with open(os.path.join(args.out_dir, "data_split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "used_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Saved model and outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
