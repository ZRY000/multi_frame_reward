ckpt='/home/juntao/Models/videocrafter/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'
PORT=$((20000 + RANDOM % 10000))

accelerate launch --multi_gpu --main_process_port $PORT scripts/main/vg_t2v_lora.py \
--seed 300 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--frames 12 \
--prompt_path '/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/prompts_vc.json' \
--gradient_accumulation_steps 8 \
--num_train_epochs 36 \
--train_batch_size 3 \
--val_batch_size 1 \
--num_val_runs 1 \
--reward_fn 'vllava' \
--decode_frame '-1' \
--hps_version 'v2.1' \
--lr 0.0002 \
--validation_steps 10 \
--lora_rank 16 \
--is_sample_preview True \
--use_wandb True \
--wandb_entity "pku_rl" \
--checkpointing_steps 5 \
--backprop_mode "last" \
--critic_type "multi-frame" \
--critic_ckp \
--critic_name "hps_clip" \
--critic_ds_size 4 \
--gamma 0.99 \
--actor_loss_type "same" \
--rg_type "draft" \
--vg_backprop 2 \
--critic_method "td-lambda" \
--lam 0.95 \
--critic_iterations 1 \
--alpha 0.2
