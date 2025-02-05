ckpt='/home/juntao/Models/videocrafter/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'
PORT=$((20000 + RANDOM % 10000))

accelerate launch --multi_gpu --main_process_port $PORT scripts/main/rg_t2v_lora_copy.py \
--seed 300 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--frames 12 \
--prompt_path '/home/juntao/Data/safe-sora/filt-prompt/new_1000_prompt.json' \
--gradient_accumulation_steps 8 \
--num_train_epochs 36 \
--train_batch_size 1 \
--val_batch_size 1 \
--num_val_runs 1 \
--reward_fn 'vllava' \
--decode_frame '-1' \
--hps_version 'v2.1' \
--lr 0.0002 \
--validation_steps 5 \
--lora_rank 16 \
--is_sample_preview True \
--use_wandb True \
--wandb_entity "pku_rl" \
--checkpointing_steps 5 \
--backprop_mode "refl" \
--mixed_precision 'bf16' \
--rew_lora_ckpt_path '/home/juntao/Projects/safe-sora/outputs/35w-3/reward-helpfulness-lora' 
# --cost_lora_ckpt_path '/home/juntao/Projects/safe-sora/outputs/vc-1/reward-harmlessness-lora/checkpoint-228'
