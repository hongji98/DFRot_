export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
msg='u5d95a100, tsvd rowonly 0.02, 20times, GPTQ, done'

# calib
echo "Generating calibration data..."
python3 scripts/generate_calibration.py --model llama-7b --nsamples 1 --output_dir calibration --disable_qk_rotation
#python3 scripts/generate_calibration.py --model /maasjfs/hf_models/llama-2-7b-chat-hf --nsamples 1 --output_dir calibration --disable_qk_rotation


# optimize rotate
echo "Optimizing rotation matrices..."
python3 scripts/optimize_procrustes_alter.py --rotate_mode hadamard --data_principle alter --alpha 100 --rotate_uniform 0
echo "$msg"