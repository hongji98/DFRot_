import os
import pickle
import sys
from functools import partial

import numpy as np
import torch
import tqdm

from utils.quant_utils import ActQuantizer
from utils.hadamard_utils import random_hadamard_matrix
import argparse
import math

def to_numpy(data):
    return data.detach().cpu().numpy()


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


def quant_func(x, n_bits, sym=True, clip_ratio=1.0):
    quantizer = ActQuantizer()
    quantizer.configure(n_bits, groupsize=-1, sym=sym, clip_ratio=clip_ratio)
    quantizer.find_params(x)
    x = quantizer(x)
    quantizer.free()
    return x


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def orthogonal_procrustes(A, B):
    C = torch.matmul(A.T, B)
    U, _, Vt = torch.linalg.svd(C, full_matrices=True)
    R = torch.matmul(U, Vt)

    return R


@torch.no_grad()
def get_best_rotate_via_procrustes(A_init, log_func, n_bits=4, steps=1000, show_init=False, clip_ratio=1.0, args=None):
    # assert is_pow2(A_init.shape[-1]), f"A.shape[-1]={A_init.shape[-1]} is not pow of two is not support!"
    shape = A_init.shape[-1]
    if show_init:
        # Initialize random hadamard matrix and randomized orthogonal matrix via QR decomposition
        hadamard = random_hadamard_matrix(shape, A_init.device).to(dtype=torch.float32)
        R = torch.randn(shape, shape)
        R_random = torch.linalg.qr(R)[0].to(A_init)
        # get init quantization error
        init_loss = torch.norm(A_init - quant_func(A_init, n_bits, clip_ratio=clip_ratio))
        log_func(f"Init Quantization Error: {init_loss:.5f}")
        # X @ hadamard
        hadamard_x1 = A_init @ hadamard - quant_func(A_init @ hadamard, n_bits, clip_ratio=clip_ratio)
        hadamard_x1_loss = torch.norm(hadamard_x1)
        log_func(f"Hadamard Quantization Error: {hadamard_x1_loss:.5f}")
        #
        random_loss = torch.norm(A_init @ R_random - quant_func(A_init @ R_random, n_bits, clip_ratio=clip_ratio))
        log_func(f"Random Quantization Error: {random_loss:.5f}")
        if args.rotate_mode == 'hadamard':
            return hadamard
        elif args.rotate_mode == 'random':
            return R_random
        else:
            raise NotImplementedError

    # Random hadamard matrix is a good beginning
    A = A_init
    R_accumulate = torch.eye(shape).to(A_init)
    best_loss = torch.norm(A - quant_func(A, n_bits=n_bits, clip_ratio=clip_ratio)) + 100

    # Procrustes Optimization
    for step in tqdm.tqdm(range(1, steps + 1), desc="(Procrustes) Optimization"):
        A_ = A @ R_accumulate
        B_ = quant_func(A_, n_bits=n_bits, clip_ratio=clip_ratio)
        R = torch.linalg.qr(orthogonal_procrustes(A_, B_))[0]
        loss = torch.norm(A @ R_accumulate @ R - quant_func(A @ R_accumulate @ R, n_bits=n_bits, clip_ratio=clip_ratio))
        log_func(f"Step: {step} Loss: {loss:.5f}")
        if loss <= best_loss and (best_loss - loss) > 0.1:
            best_loss = loss
            R_accumulate = R_accumulate @ R
        else:
            break

    return R_accumulate


def get_data(saved_path):
    # read saved
    with open(saved_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    # stack data - filter for compatible dimensions
    arrays = []
    target_dim = None
    for key, value in loaded_dict.items():
        if isinstance(value, list) and len(value) > 0:
            if hasattr(value[0], 'shape') and len(value[0].shape) == 3:
                arr = value[0].squeeze()
                if target_dim is None:
                    target_dim = arr.shape[-1]
                    print(f"Target dimension set to {target_dim} from {key}")
                if arr.shape[-1] == target_dim:
                    arrays.append(arr)
                else:
                    print(f"Skipping {key} with dimension {arr.shape[-1]} != {target_dim}")
    if not arrays:
        raise ValueError("No compatible arrays found in the data")
    output = np.concatenate(arrays, axis=0)
    print(f"Concatenated {len(arrays)} arrays, final shape: {output.shape}")
    return output

def _row_peak_metrics(X: torch.Tensor):
    # 每行 ratio_i = max(|row|) / mean(|row|)
    absX = X.abs()
    row_max = absX.max(dim=1).values
    row_mean = absX.mean(dim=1) + 1e-12
    ratio = row_max / row_mean
    peak_mean = float(ratio.mean())
    log_peak_mean = float(torch.log(ratio).mean())
    return peak_mean, log_peak_mean

@torch.no_grad()
def procrustes_step_uniform(A_cur: torch.Tensor, log_func, n_bits=4, clip_ratio=1.0, steps=1):
    device, dtype = A_cur.device, A_cur.dtype
    N, D = A_cur.shape
    R_accum = torch.eye(D, device=device, dtype=dtype)

    for s in range(1, steps + 1):
        X = A_cur @ R_accum                                  # (N, D)
        row_norms = torch.linalg.norm(X, dim=1, keepdim=True)
        c = (row_norms / math.sqrt(D))                       # (N, 1)
        signX = torch.where(X >= 0, torch.ones_like(X), -torch.ones_like(X))
        B = signX * c                                        # (N, D)

        C = X.T @ B
        U, _, Vt = torch.linalg.svd(C, full_matrices=True)
        dR = U @ Vt
        dR = torch.linalg.qr(dR)[0].to(device=device, dtype=dtype)

        R_accum = R_accum @ dR

        # 打印与原版一致的“量化误差 Loss”，并追加峰度指标
        X_now = A_cur @ R_accum
        loss = torch.norm(X_now - quant_func(X_now, n_bits=n_bits, clip_ratio=clip_ratio))
        peak_mean, log_peak_mean = _row_peak_metrics(X_now)
        log_func(f"Step: {s} Loss: {loss:.5f}  |  PeakMean={peak_mean:.5f}  LogPeakMean={log_peak_mean:.5f}")

    return R_accum



@torch.no_grad()
def procrustes_step_uniform1(
    A_cur: torch.Tensor,
    log_func,
    n_bits: int = 4,
    clip_ratio: float = 1.0,
    steps: int = 1,
):
    """
    显存友好版 Procrustes 迭代（dtype 与 A_cur 保持一致）。

    逐行构造 B 的规则：
      • |x| > 2×row_mean → sign(x) * row_mean
      • 否则             → 对称 int-4 量化-反量化（步长 = row_max / 7）
      • 最后行 2-范数归一到与 X 相同
    """
    device, dtype = A_cur.device, A_cur.dtype   # 保持原 dtype（fp32）
    N, D = A_cur.shape
    R_accum = torch.eye(D, device=device, dtype=dtype)

    # 预分配缓冲，循环内就地覆盖
    B_buf = torch.empty_like(A_cur)

    for s in range(1, steps + 1):
        # ---------- 前向旋转 ----------
        X = A_cur @ R_accum                                    # (N,D)
        absX     = X.abs()
        row_mean = absX.mean(dim=1, keepdim=True)              # (N,1)
        row_max  = absX.max(dim=1, keepdim=True).values        # (N,1)
        signX    = torch.where(X >= 0, 1.0, -1.0)

        # ---------- 构造 B ----------
        mask_large = absX > 2 * row_mean                       # bool (N,D)
        mask_small = ~mask_large

        # ① 先把“小元素”原值复制进缓冲
        B_buf.copy_(X)

        # ② 大元素 → ±row_mean
        B_buf[mask_large] = (signX * row_mean)[mask_large]

        # ③ 小元素量化-反量化（按行步长，无 broadcast）
        if mask_small.any():
            rows, cols = mask_small.nonzero(as_tuple=True)     # 索引
            step_rows  = (row_max / 7.0 + 1e-12).squeeze(1)    # (N,)
            step_sel   = step_rows[rows]                       # 对应行步长
            x_sel      = X[rows, cols]

            q_int = torch.clamp((x_sel / step_sel).round(), -7, 7)
            B_buf[rows, cols] = q_int * step_sel               # 写回

        # ---------- 行 2-范数守恒 ----------
        norm_X = torch.linalg.norm(X, dim=1, keepdim=True)     # (N,1)
        norm_B = torch.linalg.norm(B_buf, dim=1, keepdim=True)
        B_buf.mul_(norm_X / (norm_B + 1e-12))

        # ---------- Procrustes 更新 ----------
        C   = X.T @ B_buf
        U, _, Vt = torch.linalg.svd(C, full_matrices=False)    # 节省显存
        dR  = torch.linalg.qr(U @ Vt)[0].to(dtype=dtype)
        R_accum = R_accum @ dR

        # ---------- 指标打印 ----------
        X_now = A_cur @ R_accum
        loss  = torch.norm(
            X_now - quant_func(X_now, n_bits=n_bits, clip_ratio=clip_ratio)
        )
        peak_mean, log_peak_mean = _row_peak_metrics(X_now)
        log_func(
            f"Step: {s}  Loss: {loss:.5f}  |  "
            f"PeakMean={peak_mean:.5f}  LogPeakMean={log_peak_mean:.5f}"
        )

    return R_accum

@torch.no_grad()
def procrustes_step_quant(A_cur: torch.Tensor, log_func, n_bits=4, clip_ratio=1.0, steps=1):
    """
    """
    device, dtype = A_cur.device, A_cur.dtype
    D = A_cur.shape[-1]
    R_accum = torch.eye(D, device=device, dtype=dtype)

    for s in range(1, steps + 1):
        A_ = A_cur @ R_accum
        B_ = quant_func(A_, n_bits=n_bits, clip_ratio=clip_ratio)
        R = torch.linalg.qr(orthogonal_procrustes(A_, B_))[0]
        R_accum = R_accum @ R

        X_now = A_cur @ R_accum
        loss = torch.norm(X_now - quant_func(X_now, n_bits=n_bits, clip_ratio=clip_ratio))
        peak_mean, log_peak_mean = _row_peak_metrics(X_now)
        log_func(f"Step: {s} Loss: {loss:.5f}  |  PeakMean={peak_mean:.5f}  LogPeakMean={log_peak_mean:.5f}")

    return R_accum


if __name__ == "__main__":
    if not torch.cuda.is_available():
        template = ("python3 optimize_procrustes_alter.py --rotate_mode {rotate_mode} "
                    "--data_principle {data_principle} --alpha {alpha}")
        rotate_modes = ['random', 'hadamard']
        data_principles = ['alter', ]
        alphas = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        for rotate_mode in rotate_modes:
            for data_principle in data_principles:
                for alpha in alphas:
                    print(template.format(rotate_mode=rotate_mode, data_principle=data_principle, alpha=alpha))
        exit()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--data_principle', type=str, default='alter', choices=['alter', ])
    parser.add_argument('--alpha', type=float, default=100, help='massive activation importance')
    parser.add_argument('--rotate_uniform', type=int, default=0, help = "Uniform mapping rotation, 0 for unuse, 1 for use")
    args = parser.parse_args()

    device = 'cuda:0'
    # get your calibration_dir
    calibration_dir = "calibration"
    model_names = ['LLaMA-2-7B', 'LLaMA-3-8B', 'LLaMA-2-13B', "Mistral-7B", "Mistral-7B-V3", 'QWen2-7B']
    model_names = ["LLaMA-3-8B"]
    calibrate_files = ["Llama-2-7b-hf.pkl", "Meta-Llama-3-8B.pkl", "Llama-2-13b-hf.pkl", "Mistral-7B-v0.1.pkl",
                       "Mistral-7B-v0.3.pkl", 'Qwen2-7B.pkl']
    calibrate_files = ["llama-7b_qkv.pkl"]
    
    n_bits = 4
    K = 0
    if args.rotate_uniform == 1:
        K = 5
        print("Using uniform mapping rotation for first K =", K, "iterations")
    elif args.rotate_uniform == 0:
        K = 0
        print("Using quantization-aware rotation without uniform mapping for all iterations")
    else:
        raise ValueError("rotate_uniform should be 0 or 1")
    alter_num = 100+K
    clip_ratio = 1.0

    for model_name, calibrate_file in zip(model_names, calibrate_files):
        # mkdir log_dir
        data_dir = os.path.join(f"rms_norm_feature_{args.rotate_mode}_{args.data_principle}", f"{args.alpha:.0f}")
        os.makedirs(data_dir, exist_ok=True)
        log = open(os.path.join(data_dir, '{}_procrustes.txt'.format(model_name)), 'w')
        log_func = partial(print_log, log=log)
        log_func(f"{model_name}")

        # load calibration datasets
        calibrate_data_path = os.path.join(calibration_dir, calibrate_file)
        data_collects = get_data(calibrate_data_path)
        shape = data_collects.shape[-1]

        A_init = torch.from_numpy(data_collects).to(device=device, dtype=torch.float32)
        StartR = get_best_rotate_via_procrustes(A_init=A_init, n_bits=n_bits, log_func=log_func, steps=0,
                                                show_init=True, clip_ratio=clip_ratio, args=args)

        # show hadamard baseline
        _A = A_init @ StartR
        error_A_init = torch.norm(A_init - quant_func(A_init, n_bits=n_bits), dim=-1)
        R_random = torch.linalg.qr(torch.randn(shape, shape))[0].to(A_init)
        error_random = torch.norm(A_init @ R_random - quant_func(A_init @ R_random, n_bits=n_bits), dim=-1)
        R_optimize = torch.eye(shape).to(device)
        assert args.data_principle == 'alter' and alter_num >= 1

        for idx in range(alter_num):
            # Different LLM has different massive activation quantization error threshold
            if 'Mistral-7B' not in model_name:
                threshold = 8.5
                A = torch.where(error_A_init.reshape(-1, 1) < threshold, _A  * args.alpha, _A)
                num = torch.sum(error_A_init.reshape(-1, 1) < threshold)
            else:
                A = torch.where(error_random.reshape(-1, 1) < error_A_init.reshape(-1, 1), _A, _A * args.alpha)
                num = torch.sum(error_random.reshape(-1, 1) < error_A_init.reshape(-1, 1))
            optimize_max_num = 1
            log_func(
                f"{idx}-step the data shape is: {A.shape} {num}")
            log_func(
                f"Initialize: {torch.norm(A @ R_optimize - quant_func(A @ R_optimize, n_bits, True, clip_ratio)):.5f}")
            if(0):
                dR = get_best_rotate_via_procrustes(A @ R_optimize, log_func, n_bits=n_bits,
                steps=optimize_max_num, clip_ratio=clip_ratio, args=args
            )
            else:
                A_cur = A @ R_optimize
                if idx < K:
                    dR = procrustes_step_uniform(
                        A_cur, log_func, n_bits=n_bits, clip_ratio=clip_ratio, steps=optimize_max_num
                    )
                else:
                    dR = procrustes_step_quant(
                        A_cur, log_func, n_bits=n_bits, clip_ratio=clip_ratio, steps=optimize_max_num
                    )
            R_optimize = R_optimize @ dR
            R_optimize = torch.linalg.qr(R_optimize)[0]

        # Get optimized R
        R_optimize = StartR @ R_optimize
        log_func(f"Sum of Diag(R @ R.T):  {torch.sum(torch.diag(R_optimize @ R_optimize.T))}")

        loss = torch.norm(A_init @ R_optimize - quant_func(A_init @ R_optimize, n_bits=n_bits, clip_ratio=clip_ratio))
        log_func(f"Final Quantization Error: {loss:.5f}")
        peak_mean, log_peak_mean = _row_peak_metrics(A_init @ R_optimize)
        log_func(f"Final PeakMean={peak_mean:.5f} LogPeakMean={log_peak_mean:.5f}")
        # save R to numpy
        R_optimize = to_numpy(R_optimize)
        save_path = os.path.join(data_dir, "u"+f"{K}"+"d"+f"{alter_num}"+"a4t100"+"b" + f"-{n_bits}nn.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, R_optimize)
        print(f"Saved to {save_path}")
        log.close()
