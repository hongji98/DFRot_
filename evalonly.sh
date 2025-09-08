export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
msg= '0 svd, GPTQ, done'
echo "$msg"
echo "Evaluating model with rotation..."
python3 main.py --model llama-7b --rotate --svd\
 --svd_type "r" --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 --w_clip --v_groupsize 128 --k_groupsize 128 \
  --a_asym --k_asym --v_asym --rotate_mode orthogonal_procrustes \
  --indices_path rms_norm_feature_hadamard_alter/100/u0d100a4t100b-4nn.npy --fp32_had --seed 0 --lm_eval
echo "$msg"