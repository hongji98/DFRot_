export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
msg='u5d95a1, tsvd rowonly 0.02, 20times, GPTQ, done'

# calib
echo "Generating calibration data..."
#python3 scripts/generate_calibration.py --model /maasjfs/hf_models/llama-2-7b-chat-hf --nsamples 1 --output_dir calibration --disable_qk_rotation
#python3 scripts/generate_calibration.py --model /maasjfs/hf_models/meta-llama3.1/Meta-Llama-3.1-8B-Instruct --nsamples 1 --output_dir calibration --disable_qk_rotation
python3 scripts/generate_calibration.py --model Meta-Llama-3-8B-Instruct --nsamples 1 --output_dir calibration --disable_qk_rotation


# optimize rotate
echo "Optimizing rotation matrices..."
python3 scripts/optimize_procrustes_alter.py --rotate_mode hadamard --data_principle alter --alpha 100
echo "$msg"

# eval
echo "Evaluating model with rotation..."
python3 main.py --model Meta-Llama-3-8B-Instruct --rotate --svd \
 --svd_type "r"  --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 --w_clip --v_groupsize 128 --k_groupsize 128 \
  --a_asym --k_asym --v_asym --rotate_mode orthogonal_procrustes \
  --indices_path rms_norm_feature_hadamard_alter/100/LLaMA-3-8B-4.npy --fp32_had --seed 0 --lm_eval
echo "$msg"