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

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling, rg_batch_ddim_sampling
from utils.utils import instantiate_from_config
from lvdm.models.reward_copy import VllavaReward, InternV2Reward, CLIPReward, HPSReward

import peft
from peft import PeftModel
import torchvision
import torch
import torch.nn as nn
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
import torch.nn.functional as F
import ast
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
# import ipdb
# st = ipdb.set_trace


def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)
    
logger = get_logger(__name__, log_level="INFO") # get logger for current module

def tuple_type(s):
    if isinstance(s, tuple):
        return s
    value = ast.literal_eval(s)
    if isinstance(value, tuple):
        return value
    raise TypeError("Argument must be a tuple")

class PromptDataset(Dataset):
    def __init__(self, prompts, sources):
        self.prompts = prompts
        self.sources = sources

    def __len__(self):
        assert len(self.prompts) == len(self.sources)
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        source = self.sources[idx]
        return {
            'prompt': prompt,
            'source': source,
        }

# Function to get sampler for distributed training
def get_sampler(dataset, distributed=False):
    if distributed:
        return DistributedSampler(dataset)
    else:
        return None

# Function to create dataloader
def get_dataloader(prompts, sources, batch_size, distributed=False):
    dataset = PromptDataset(prompts, sources)
    sampler = get_sampler(dataset, distributed)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
    return dataloader

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

# to convert string to boolean in argparse
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
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup_ratio")
    parser.add_argument(
        "--reward_train_processes",
        type=tuple_type,
        default=(0, 1, 2, 3, 4, 5),
        help="Process idx that are used to maximize text-img reward fn.",
    )
    parser.add_argument(
        "--video_rm_train_processes",
        type=tuple_type,
        default=(6, 7),
        help="Process idx that are used to maximize text-video reward fn.",
    )
    parser.add_argument("--unet_ckpt_path", type=str, default=None, help="unet checkpoint path")
    parser.add_argument("--unet_time_cond_proj_dim", type=int, default=256, help="The dimension of the guidance scale embedding in the U-Net")


    return parser

def should_sample(global_step, validation_steps, is_sample_preview):
    return (global_step % validation_steps == 0 or global_step ==1)  \
    and is_sample_preview

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

@dataclass
class ModelArguments:
    model_name_or_path: str = "/home/juntao/Models/LanguageBind/Video-LLaVA-7B"
    version: str = "v1"
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = True
    vision_tower: str = None
    mm_vision_select_layer: int = -2  # default to the last layer
    pretrain_mm_mlp_adapter: str = "/home/juntao/Models/LanguageBind/Video-LLaVA-Pretrain-7B/mm_projector.bin"
    mm_projector_type: str = "mlp2x_gelu"
    mm_use_im_start_end: bool = False
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: str = None

    # ===================================================================
    image_tower: str = "LanguageBind/LanguageBind_Image"
    video_tower: str = "LanguageBind/LanguageBind_Video_merge"
    # ===================================================================

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster

REWARD_NORMALIZE = {
    "vllava_score": {
        "mean": 2.465740198516846,
        "std": 2.8097259558435366
    },
    "internv2_score": {
        "mean": 0.4162220517873764,
        "std": 0.04686144238576121
    },
    "clip_score": {
        "mean": 0.42187142968177793,
        "std": 0.04124788141767662
    },
    "hps_score": {
        "mean": 0.3088359375,
        "std": 0.03172493722637392
    }
}

def run_training(args, **kwargs):
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

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.use_wandb:
        if accelerator.is_main_process:
            import wandb

            wandb_args = {}
            

            if args.wandb_entity != '':
                wandb_args['entity'] =  args.wandb_entity

            if args.debug:
                wandb_args['mode'] = "disabled"

            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            exp_name = f"vc_rg_{args.backprop_mode}_bsl"
            task_name='distilled_intern_clip'
            wandb_args['name'] = task_name
            # wandb_args['name'] = f"{exp_name}-{current_time}"

            opt_dict = vars(args)   # convert args to dict
            accelerator.init_trackers("Value-gradient", config=opt_dict, init_kwargs={"wandb": wandb_args})
            output_dir = create_output_folders(args.project_dir, wandb.run.name)    # all processes will create the same output folder
            # convert output_dir to broadcastable tensor, so that it can be broadcasted to all processes
            output_dir_broadcast = [output_dir]

        else:
            output_dir_broadcast = [None]

        # convert output_dir back to str
        dist.broadcast_object_list(output_dir_broadcast, src=0)    # broadcast the output_dir to all processes
        output_dir = output_dir_broadcast[0]
        print(f"+++++++++++++++++++output_dir: {output_dir}+++++++++++++++++++++++++++++++++")
        
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)

    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)

    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = args.unet_time_cond_proj_dim
    unet = instantiate_from_config(unet_config)
    unet.load_state_dict(
        torch.load(args.unet_ckpt_path, map_location=accelerator.device, weights_only=True)
    )
    model.model.diffusion_model = unet
    
    logger.info("Saving checkpointing....")
    # resave_the model
    save_path = os.path.join("/home/juntao/Models/distillation/full-model", "model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
        
    logger.info("Saving checkpoing end")
    


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@VADER-VideoCrafter: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_training(args)
