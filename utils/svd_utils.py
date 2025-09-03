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
        # Default: use 128 or max_rank
        return min(128, max_rank)


def svd_fwrd(model, dev, args):
    """SVD-based weight quantization with low-rank + residual decomposition"""
    assert args.w_groupsize == -1, "Groupwise quantization not supported with SVD yet"

    layers = model.model.layers
    torch.cuda.empty_cache()
    quantizers = {}

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

                # Compute residual
                W_lowrank = L1 @ L2
                residual = W - W_lowrank

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
                linear.weight.data.copy_(residual_quant)
                linear.register_parameter("L1", nn.Parameter(L1))
                linear.register_parameter("L2", nn.Parameter(L2))

                # Reconstruct quantized weight
                W_final = (W_lowrank + residual_quant).to(linear.weight.dtype)

                # Compute metrics for logging
                energy_retained = (S[:rank]**2).sum() / (S**2).sum() if S.numel() > 0 else 1.0
                mse = ((W - W_final)**2).mean().item()

                print(f"[{m}x{n}→r{rank}, E:{energy_retained:.3f}, MSE:{mse:.2e}]")

                # Store quantizer and metadata
                quantizers[f"{i}.{name}"] = {
                    'rank': rank,
                    'bits': layer_weight_bits,
                    'energy': energy_retained.item() if torch.is_tensor(energy_retained) else energy_retained,
                    'mse': mse,
                    'quantizer': quantizer
                }

                # Cleanup
                del W, U, S, Vh, L1, L2, W_lowrank, residual, residual_quant
                torch.cuda.empty_cache()

        # Move layer back to CPU to save memory
        layers[i] = layer.cpu()
        misc.cleanup_memory(verbos=False)

    return quantizers

