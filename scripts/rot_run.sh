export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4

# calib
python3 scripts/generate_calibration.py --model /nvme_data/hf_models/meta-llama/Llama-3.1-8B-Instruct --nsamples 1 --output_dir calibration --disable_qk_rotation

# optimize rotate
python3 scripts/optimize_procrustes_alter.py --rotate_mode hadamard --data_principle alter --alpha 100

# eval
python3 main.py --model /nvme_data/hf_models/meta-llama/Llama-3.1-8B-Instruct --no-rotate \
  --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 --w_clip --v_groupsize 128 --k_groupsize 128 \
  --a_asym --k_asym --v_asym --rotate_mode orthogonal_procrustes \
  --indices_path rms_norm_feature_hadamard_alter/100/LLaMA-3-8B-4.npy --fp32_had --seed 0 --lm_eval
