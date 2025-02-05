ckpt='/home/juntao/Models/videocrafter/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'
PORT=$((20001 + RANDOM % 10000))

accelerate launch --multi_gpu --main_process_port $PORT scripts/main/eval_with_score.py \
--seed 300 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--frames 12 \
--val_batch_size 1 \
--validation_steps 5 \
--lora_rank 16 \
--is_sample_preview True \
--num_val_runs 5 \
--prompt_path "/home/juntao/Data/safe-sora/filt-prompt/new_1000_prompt.json" \
--rew_lora_ckpt_path '/home/juntao/Projects/safe-sora/outputs/vc-2/reward-helpfulness-lora/checkpoint-969' \
# --lora_ckpt_path "/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/project_dir/vc_rg_refl-2024-12-20-15-14-11/peft_model_280.pt"
