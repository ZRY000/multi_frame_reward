import argparse, os, sys, glob, yaml, math, random
sys.path.append('../')   # setting path to get Core and assets

import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything
import json
from dataclasses import dataclass, field
from typing import Any
import hashlib

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling, rg_batch_ddim_sampling
from utils.utils import instantiate_from_config
from lvdm.models.reward import VllavaReward, InternV2Reward, CLIPReward, HPSReward

import peft
from peft import PeftModel
import torchvision
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers.utils import ContextManagers
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import gather_object
import torch.distributed as dist
import logging
import gc
from PIL import Image
import io
import albumentations as A
from huggingface_hub import snapshot_download
# import ipdb
# st = ipdb.set_trace
import deepspeed
from copy import deepcopy

logger = get_logger(__name__, log_level="INFO") # get logger for current module

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def create_output_folders(output_dir, run_name):
    out_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    return out_dir

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    ## for training
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--val_batch_size", type=int, default=1, help="batch size for validation")
    parser.add_argument("--num_val_runs", type=int, default=1, help="total number of validation samples = num_val_runs * num_gpus * num_val_batch")
    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training")
    parser.add_argument("--reward_fn", type=str, default="aesthetic", help="reward function: 'aesthetic', 'hps', 'aesthetic_hps', 'pick_score', 'rainy', 'snowy', 'objectDetection', 'actpred', 'compression'")
    parser.add_argument("--compression_model_path", type=str, default='../assets/compression_reward.pt', help="compression model path") # The compression model is used only when reward_fn is 'compression'
    # The "book." is for grounding-dino model . Remember to add "." at the end of the object name for grounding-dino model. 
    # But for yolos model, do not add "." at the end of the object name. Instead, you should set the object name to "book" for example.
    parser.add_argument("--target_object", type=str, default="book", help="target object for object detection reward function")
    parser.add_argument("--detector_model", type=str, default="yolos-base", help="object detection model", 
                            choices=["yolos-base", "yolos-tiny", "grounding-dino-base", "grounding-dino-tiny"])
    parser.add_argument("--hps_version", type=str, default="v2.1", help="hps version: 'v2.0', 'v2.1'")
    parser.add_argument("--prompt_fn", type=str, default="hps_custom", help="prompt function")
    parser.add_argument("--nouns_file", type=str, default="simple_animals.txt", help="nouns file")
    parser.add_argument("--activities_file", type=str, default="activities.txt", help="activities file")
    parser.add_argument("--num_train_epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=10000, help="max training steps")
    parser.add_argument("--backprop_mode", type=str, default="last", help="backpropagation mode: 'last', 'rand', 'specific'")   # backprop_mode != None also means training mode for batch_ddim_sampling
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default='fp16', help="mixed precision training: 'no', 'fp8', 'fp16', 'bf16'")
    parser.add_argument("--logger_type", type=str, default="wandb", help="logger type: 'wandb', 'tensorboard'")
    parser.add_argument("--project_dir", type=str, default="./project_dir", help="project directory")
    parser.add_argument("--validation_steps", type=int, default=1, help="The frequency of validation, e.g., 1 means validate every 1*accelerator.num_processes steps")
    parser.add_argument("--checkpointing_steps", type=int, default=1, help="The frequency of checkpointing")
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="use wandb for logging")
    parser.add_argument("--wandb_entity", type=str, default="", help="wandb entity")
    parser.add_argument("--debug", type=str2bool, default=False, help="debug mode")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max gradient norm")
    parser.add_argument("--use_AdamW8bit", type=str2bool, default=False, help="use AdamW8bit optimizer")
    parser.add_argument("--is_sample_preview", type=str2bool, default=True, help="sample preview during training")
    parser.add_argument("--decode_frame", type=str, default="-1", help="decode frame: '-1', 'fml', 'all', 'alt'") # it could also be any number str like '3', '10'. alt: alternate frames, fml: first, middle, last frames, all: all frames. '-1': random frame
    parser.add_argument("--inference_only", type=str2bool, default=False, help="only do inference")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--prompt_path", type=str, default=None, help="path to the prompts")
    parser.add_argument("--rew_lora_ckpt_path", type=str, default=None, help="Reward Model LoRA checkpoint path")
    parser.add_argument("--cost_lora_ckpt_path", type=str, default=None, help="Cost Model LoRA checkpoint path")
    

    return parser

def main(args, **kwargs):
    ## ---------------------step 1: accelerator setup---------------------------
    accelerator = Accelerator(                                                  # Initialize Accelerator
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger_type,
        project_dir=args.project_dir
    )
    output_dir = args.project_dir

    validation_steps = args.validation_steps * args.gradient_accumulation_steps         # number of steps to run validation for
    checkpointing_steps = args.checkpointing_steps * args.gradient_accumulation_steps   # Saves a model every nth step.

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"eval-vc-{current_time}"
    output_dir = os.path.join(output_dir, exp_name)
    output_dir_broadcast = [output_dir] # for broadcasting
    dist.broadcast_object_list(output_dir_broadcast, src=0)    # broadcast the output_dir to all processes
    output_dir = output_dir_broadcast[0]
    print(f"+++++++++++++++++++output_dir: {output_dir}+++++++++++++++++++++++++++++++++")

    ## ------------------------step 2: model config-----------------------------
    # download the checkpoint for VideoCrafter2 model
    # ckpt_dir = args.ckpt_path.split('/')    # args.ckpt='checkpoints/base_512_v2/model.ckpt' -> 'checkpoints/base_512_v2'
    # ckpt_dir = '/'.join(ckpt_dir[:-1])
    # snapshot_download(repo_id='VideoCrafter/VideoCrafter2', local_dir =ckpt_dir)
    
    # load the model
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)

    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)

    # step 2.1: add LoRA using peft
    config = peft.LoraConfig(
            r=args.lora_rank,
            lora_alpha=2*args.lora_rank,
            target_modules=["to_k", "to_v", "to_q"],        # only diffusion_model has these modules
            lora_dropout=0.01,
        )
    
    peft_model = peft.get_peft_model(model, config)

    peft_model.print_trainable_parameters()

    # load the pretrained LoRA model
    if args.lora_ckpt_path is not None:
        # load the pretrained LoRA model
        peft.set_peft_model_state_dict(peft_model, torch.load(args.lora_ckpt_path))

    rew_model = VllavaReward(accelerator.device, "/home/juntao/Models/LanguageBind/Video-LLaVA-7B", use_grad_checkpoint=True)
    rew_model.model.model = PeftModel.from_pretrained(
        model=rew_model.model.model,
        model_id=args.rew_lora_ckpt_path,
        adapter_name="reward"
    )
    rew_model.model.model.load_adapter(
        model_id=args.cost_lora_ckpt_path,
        adapter_name="cost"
    )

    # ==== load video tower ====
    if rew_model.model.config.mm_video_tower is not None:
        video_tower =rew_model.model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_tower.to(device=accelerator.device, dtype=torch.bfloat16)

    rew_model.load_score(
        score_head_path=os.path.join(args.rew_lora_ckpt_path, 'score_head.pt'),
        score_name="reward"
    )
    rew_model.load_score(
        score_head_path=os.path.join(args.cost_lora_ckpt_path, 'score_head.pt'),
        score_name="cost"
    )

    internv2 = InternV2Reward(rm_ckpt_dir='/home/juntao/Models/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt')
    clip = CLIPReward(precision='bf16')
    hps = HPSReward(device=accelerator.device, hps_version="v2.1")
    
    peft_model = accelerator.prepare(peft_model)

    # sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    # latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = peft_model.module.temporal_length if args.frames < 0 else args.frames
    channels = peft_model.module.channels

    # == load prompts ==
    with open(args.prompt_path, "r") as f:
        # prompts = [line.strip() for line in f.readlines()]
        prompts_cfg = json.load(f)
        # prompts = []
        # sources = []
        # for prompt_item in prompts_cfg:
        #     if prompt_item["source"] != "unsafe":
        #         prompts.append(prompt_item["prompt_text"])
        #         sources.append(prompt_item["source"])
        # assert len(prompts) == 500
        prompts = [prompt_item["prompt_text"] for prompt_item in prompts_cfg]
        sources = [prompt_item["source"] for prompt_item in prompts_cfg]
        assert len(prompts) == 1000
    assert isinstance(prompts, list)
    if accelerator.is_main_process:
        print(prompts[600])
    video_output_dir=os.path.join(output_dir, "videos")
    os.makedirs(video_output_dir, exist_ok=True)
    # == prepare arguments ==
    
    # == prepare inference ==
    if accelerator.is_main_process:
        start = time.time()
    ## Inference Step 5: generate new validation videos
    with accelerator.autocast():
        with torch.no_grad():
            # for dimension in dimension_list:
            #     # read prompt list
            #     with open(f'/home/juntao/Projects/zry/value-gradient/assets/vbench_prompts/prompts_per_dimension/{dimension}.txt', 'r') as f:
            #         prompt_list = f.readlines()
            prompt_list = [prompt.strip() for prompt in prompts]
            # sample 5 videos for each prompt

            for index in range(args.num_val_runs):
                random.seed(args.seed + index)
                torch.manual_seed(args.seed + index)
                prompt_idx = list(range(len(prompt_list)))
                with accelerator.split_between_processes(prompt_idx) as val_idx:
                    if accelerator.is_main_process:
                        verbose = 1
                        progress_bar = tqdm(list(enumerate(val_idx)), desc=f"Inference loop {index}", disable=not verbose)
                    val_prompts = [prompt_list[i] for i in val_idx]
                    val_sources = [sources[i] for i in val_idx]
                    results = []
                    for j, val_prompt, val_source in enumerate(zip(val_prompts, val_sources)):
                        val_prompt = [val_prompt]
                        assert len(val_prompt) == args.val_batch_size

                        val_batch_size = len(val_prompt)
                        noise_shape = [val_batch_size, channels, frames, h, w]
                        
                        fps = torch.tensor([args.fps]*val_batch_size).to(accelerator.device).long()

                        
                        text_emb = peft_model.module.get_learned_conditioning(val_prompt).to(accelerator.device)

                        if args.mode == 'base':
                            cond = {"c_crossattn": [text_emb], "fps": fps}
                        else:   # TODO: implement i2v mode training in the future
                            raise NotImplementedError

                        batch_samples = batch_ddim_sampling(peft_model.module, cond, noise_shape, args.n_samples, \
                                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, decode_frame=args.decode_frame, **kwargs)
                        video_frames_ = batch_samples.squeeze(1)
                        rew_model.model.model.set_adapter("reward")
                        rew_model.set_score("reward")
                        vllava_reward = rew_model(video_frames_, val_prompt)
                        rew_model.model.model.set_adapter("cost")
                        rew_model.set_score("cost")
                        cost = rew_model(video_frames_, val_prompt)

                        start_index = random.randint(0, video_frames_.shape[2]-1-4)
                        internv2_reward = internv2(video_frames_[:,:,start_index:start_index + 4,:,:].permute(0,2,1,3,4), val_prompt)

                        frame_index = random.randint(0, video_frames_.shape[2]-1)
                        clip_reward = clip(video_frames_[:,:,frame_index,:,:], val_prompt)
                        hps_reward = hps(video_frames_[:,:,frame_index,:,:], val_prompt)

                        dir_name = video_output_dir
                        os.makedirs(dir_name, exist_ok=True)
                        filenames = [f"{index}_{accelerator.local_process_index}_{j+1:04d}_{time.time()}"]
                        filenames = [hashlib.md5(filenames[0].encode()).hexdigest()]
                        sample_config = {
                            'video_path': filenames[0]+'.mp4',
                            'prompt_text': val_prompt[0],
                            'source': val_source,
                            'vllava_score': vllava_reward.item(),
                            'internv2_score': internv2_reward.item(),
                            'clip_score': clip_reward.item(),
                            'hps_score': hps_reward.item(),
                            'safety_score': cost.item(),
                        }
                        # filenames = [f'{val_prompt[0]}-{index}' for id in range(batch_samples.shape[0])] # from 0 to batch size, n is the index of the batch
                        save_videos(batch_samples, dir_name, filenames, fps=args.savefps)
                        results.append(sample_config)
                        if accelerator.is_main_process:
                            progress_bar.update(1)
                        del batch_samples
                        torch.cuda.empty_cache()
                # collect inference results from all the GPUs
                results_gathered=gather_object(results)
                if accelerator.is_main_process:
                    config_path = os.path.join(output_dir, f"config{index}.json")
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(results_gathered, f, indent=4)
                gc.collect()
                if accelerator.is_main_process:
                    logger.info(f"Inference step {index} finished in {time.time() - start:.2f}s")
    
    dist.barrier()
    if accelerator.is_main_process:
        all_config = []
        prompt_file = {}
        for i in range(args.num_val_runs):
            config_path = os.path.join(output_dir, f"config{i}.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for item in config:
                all_config.append(item)
                prompt_file[item['video_path']] = item['prompt_text']
            os.remove(config_path)

        mean_std = {}
        mean_std["info"] = "The mean and std of safety score is calculated over 500*num_val_runs samples generated by 500 unsafe prompts; "
        mean_std["info"] += "the mean and std of other 4 scores is calculated over 500*num_val_runs samples generated by 500 safe prompts"
        names = ['vllava_score', 'internv2_score', 'clip_score', 'hps_score']
        # calculate mean and std for safe prompts
        for name in names:
            # score_list = [x[name] for x in all_config]
            score_list = []
            for x in all_config:
                if x["source"] == "dvg" or "safe":
                    score_list.append(x[name])
            assert len(score_list) == 500 * args.num_val_runs, f"actual samples: {len(score_list)}"
            mean = sum(score_list) / len(score_list)
            variance = sum((x - mean) ** 2 for x in score_list) / len(score_list)  
            std = math.sqrt(variance)
            mean_std[name] = {'mean': mean, 'std': std}
        # calculate mean and std for unsafe prompts
        score_list = []
        for x in all_config:
            if x["source"] == "unsafe":
                score_list.append(x["safety_score"])
        assert len(score_list) == 500 * args.num_val_runs, f"actual samples: {len(score_list)}"
        mean = sum(score_list) / len(score_list)
        variance = sum((x - mean) ** 2 for x in score_list) / len(score_list)  
        std = math.sqrt(variance)
        mean_std["safety_score"] = {'mean': mean, 'std': std}

        eval_info={
            'prompts': args.prompt_path,
            'validation_runs': args.num_val_runs,
            'total_samples': len(prompt_file),
            'total_prompts': len(prompts),
            'checkpoint_info': args.lora_ckpt_path,
            'mean_std': mean_std,
            'config': all_config,
            'prompt_file': prompt_file,
        }

        with open(os.path.join(output_dir, "eval_info.json"), "w", encoding="utf-8") as f:
            json.dump(eval_info, f, indent=4)
        # with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        #     json.dump(all_config, f, indent=4)
        # with open(os.path.join(output_dir, "prompt_file.json"), "w", encoding="utf-8") as f:
        #     json.dump(prompt_file, f, indent=4)
        logger.info(f"Custom evaluation finished in {time.time() - start:.2f}s")

        
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)
