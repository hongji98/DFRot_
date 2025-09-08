import torch
import tqdm

import misc
from utils import quant_utils
from torch import nn


def compute_rank(S, args, m, n):
    """Compute rank based on energy retention or fixed rank"""
    max_rank = min(m, n)

    if hasattr(args, 'svd_rank') and args.svd_rank > 0:
        # Use fixed rank
        return min(args.svd_rank, max_rank)
    elif hasattr(args, 'svd_energy') and args.svd_energy > 0:
        # Compute rank from cumulative energy
        energy_cumsum = torch.cumsum(S**2, dim=0)
        total_energy = energy_cumsum[-1]
        target_energy = args.svd_energy * total_energy
        rank = torch.searchsorted(energy_cumsum, target_energy).item() + 1
        return min(rank, max_rank)
    else:
        # Default: use 32 or max_rank
        return min(32, max_rank)

# ==== NEW: helper for mmratio & quanterr =====================================
def _mmratio_quanterr(mat: torch.Tensor, denom_norm: float | None = None):
    """
    """
    # -------- mmratio ----------
    row_max  = mat.abs().max(dim=1).values         # (m,)
    row_mean = mat.abs().mean(dim=1) + 1e-12       # avoid 0-div
    mmratio  = (row_max / row_mean).mean().item()

    # -------- row-wise int4 symmetric quant ----------
    scale    = row_max / 7.0 + 1e-12               # 4-bit signed: [-8, 7]
    q        = torch.clamp((mat / scale.unsqueeze(1)).round(), -8, 7)
    deq      = q * scale.unsqueeze(1)

    diff_norm  = (mat - deq).norm()
    base_norm  = denom_norm if denom_norm is not None else mat.norm() + 1e-12
    quanterr   = (diff_norm / base_norm).item()

    return mmratio, quanterr
# =============================================================================


def tune_row_fp32(
    L1: torch.Tensor,
    L2: torch.Tensor,
    R: torch.Tensor,
    i: int,
    k_max: int = 10,
    k: float = 4.0,
    alpha: float = 0.02,
) -> None:
    if alpha < 0.01:
        return

    r = R[i]
    for _ in range(k_max):
        abs_mean = r.abs().mean()
        tau_stop = k * abs_mean.item()
        idx_max = torch.argmax(r.abs()).item()
        r_max = r[idx_max].item()

        if abs(r_max) <= tau_stop:
            break

        tau_target = 2.0 * abs_mean.item() * (1 if r_max > 0 else -1)
        v = L2[:, idx_max]
        denom = v.dot(v).item()

        if denom:
            delta_u = -(tau_target - r_max) / denom * v
            r_pred = r - alpha * (delta_u @ L2)
            if r_pred.abs().max() < r.abs().max():
                L1[i] += alpha * delta_u
                r.copy_(r_pred)
                continue

        col = torch.argmax(v.abs()).item()
        if v[col]:
            delta_u = torch.zeros_like(L1[i])
            delta_u[col] = (tau_target - r_max) / v[col]
            r_pred = r - alpha * (delta_u @ L2)
            if r_pred.abs().max() < r.abs().max():
                L1[i] += alpha * delta_u
                r.copy_(r_pred)
                continue
        #tune_row_fp32(L1, L2, R, i, k_max, k, alpha / 2)
        break


def tune_col_fp32(
    L1: torch.Tensor,
    L2: torch.Tensor,
    R: torch.Tensor,
    j: int,
    k_max: int = 10,
    k: float = 4.0,
    alpha: float = 0.02,
) -> None:
    """
    逐列微调 (low-rank) 以降低残差 R 的 max-abs 峰值。
    与 tune_row_fp32 完全对称：这里改 L2[:, j]，影响的是整列 R[:, j]。
    """
    if alpha < 0.01:
        return

    c = R[:, j]              # 这一列残差，shape = (m,)
    for _ in range(k_max):
        abs_mean = c.abs().mean()
        tau_stop = k * abs_mean.item()

        idx_max = torch.argmax(c.abs()).item()  # 在这一列里找最大残差行号
        c_max = c[idx_max].item()

        if abs(c_max) <= tau_stop:
            break

        # 目标幅值 (同 row 版)
        tau_target = 2.0 * abs_mean.item() * (1 if c_max > 0 else -1)

        u = L1[idx_max]       # 取影响最大行对应的 L1 行向量
        denom = u.dot(u).item()

        if denom:
            # 方案 1：用 full vector 校正
            delta_v = -(tau_target - c_max) / denom * u
            c_pred = c - alpha * (L1 @ delta_v)  # 注意列更新公式
            if c_pred.abs().max() < c.abs().max():
                L2[:, j] += alpha * delta_v
                c.copy_(c_pred)
                continue

        # 方案 2：只用单个元素调
        row = torch.argmax(u.abs()).item()
        if u[row]:
            delta_v = torch.zeros_like(L2[:, j])
            delta_v[row] = (tau_target - c_max) / u[row]
            c_pred = c - alpha * (L1 @ delta_v)
            if c_pred.abs().max() < c.abs().max():
                L2[:, j] += alpha * delta_v
                c.copy_(c_pred)
                continue
        # 如果两种方案都没带来改进，则减小步长或退出
        #tune_col_fp32(L1, L2, R, j, k_max, k, alpha / 2)
        break

def svd_fwrd(model, dev, args):
    """SVD-based weight quantization with low-rank + residual decomposition"""
    assert args.w_groupsize == -1, "Groupwise quantization not supported with SVD yet"

    layers = model.model.layers
    torch.cuda.empty_cache()
    mm_before_sum, qerr_before_sum = 0.0, 0.0
    mm_after_sum,  qerr_after_sum  = 0.0, 0.0
    n_mats = 0
    coltune = False
    rowtune = False
    if args.svd_type == "n":
        coltune = False
        rowtune = False
        print("No tuning!")
    elif args.svd_type == "r":
        coltune = False
        rowtune = True
        print("Row tuning!")
    elif args.svd_type == "c":
        coltune = True
        rowtune = False
        print("Column tuning!")
    elif args.svd_type == "b":
        coltune = True
        rowtune = True
        print("Both tuning!")
    else:
        raise NotImplementedError
    for i in tqdm.tqdm(range(len(layers)), desc="(SVD Quant.) Layers"):
        layer = layers[i].to(dev)
        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            # Skip lm_head
            if "lm_head" in name:
                continue

            # Determine bits
            layer_weight_bits = args.w_bits
            layer_weight_sym = not args.w_asym
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8

            # Skip if no quantization needed
            if layer_weight_bits >= 16:
                continue

            print(f"  {name}", end=" ", flush=True)

            # Get the linear layer
            linear = subset[name]

            # SVD decomposition following specified approach
            with torch.no_grad():
                W = linear.weight.float()
                m, n = W.shape
                mm_b, qe_b = _mmratio_quanterr(W)
                mm_before_sum  += mm_b
                qerr_before_sum += qe_b
                # Compute rank
                # Preliminary SVD to get singular values for rank selection
                if hasattr(args, 'svd_energy') and args.svd_energy > 0 and (not hasattr(args, 'svd_rank') or args.svd_rank <= 0):
                    _, S_temp, _ = torch.linalg.svd(W, full_matrices=False)
                    rank = compute_rank(S_temp, args, m, n)
                    del S_temp
                else:
                    rank = compute_rank(None, args, m, n)

                rank = min(rank, *W.shape)
                                    

                # Full SVD
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                # Store low-rank factors (following specified approach)
                L1 = U[:, :rank] * S[:rank].unsqueeze(0)  # U * S
                L2 = Vh[:rank]                            # Vh            
                tmpresidual = W - (L1 @ L2)
                #tmpmax = tmpresidual.abs().max().item()
                if coltune:
                    for ite in range(L2.shape[1]):
                        tune_col_fp32(L1,L2, tmpresidual, ite)
                if rowtune:
                    for ite1 in range(L1.shape[0]):
                        tune_row_fp32(L1,L2, tmpresidual, ite1)
                                    
                #print(f"[Tuning changed maxval from {tmpmax} to {tmpresidual.abs().max().item()}]")
                # Compute residual
                W_lowrank = L1 @ L2
                residual = W - W_lowrank
                mm_a, qe_a = _mmratio_quanterr(residual, denom_norm=W.norm())
                mm_after_sum  += mm_a
                qerr_after_sum += qe_a
                n_mats += 1
                linear.weight.data.copy_(residual)
                w_dtype = linear.weight.dtype
                linear.register_parameter("L1", nn.Parameter(L1.type(w_dtype)))
                linear.register_parameter("L2", nn.Parameter(L2.type(w_dtype)))

                # Quantize residual using existing WeightQuantizer
                quantizer = quant_utils.WeightQuantizer()
                quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip
                )
                quantizer.find_params(residual)
                residual_quant = quantizer.quantize(residual)

                # Reconstruct quantized weight
                W_final = (W_lowrank + residual_quant).type(linear.weight.dtype)

                # Compute metrics for logging
                energy_retained = (S[:rank]**2).sum() / (S**2).sum() if S.numel() > 0 else 1.0
                mse = ((W - W_final)**2).mean().item()

                print(f"[{m}x{n}→r{rank}, mmratio {mm_b:.2f}->{mm_a:.2f}, quanterr {qe_b:.3e}->{qe_a:.3e}")

                # Cleanup
                del W, U, S, Vh, L1, L2, W_lowrank, residual
                torch.cuda.empty_cache()

        # Move layer back to CPU to save memory
        layers[i] = layer.cpu()
        misc.cleanup_memory(verbos=False)
    if n_mats:
        print("\n=== Average mmratio / quanterr over", n_mats, "matrices ===")
        print(f"Before SVD : mmratio={mm_before_sum / n_mats:.3f}," f"quanterr={qerr_before_sum / n_mats:.3e}")
        print(f"After  SVD : mmratio={mm_after_sum  / n_mats:.3f}," f"quanterr={qerr_after_sum  / n_mats:.3e}")
