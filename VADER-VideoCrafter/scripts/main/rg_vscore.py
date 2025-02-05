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
            task_name='intern_clip_cost'
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

    # model.first_stage_model.requires_grad_(False)
    # model.cond_stage_model.requires_grad_(False)
    # model.model.diffusion_model.requires_grad_(True).train()

    # convert first_stage_model and cond_stage_model to torch.float16 if mixed_precision is True # TODO: check if this is correct
    if args.mixed_precision != 'no':
        model.first_stage_model = model.first_stage_model.to(torch.bfloat16)
        model.cond_stage_model = model.cond_stage_model.to(torch.bfloat16)

    # step 2.1: add LoRA using peft
    config = peft.LoraConfig(
            r=args.lora_rank,
            lora_alpha=2*args.lora_rank,
            target_modules=["to_k", "to_v", "to_q"],        # only diffusion_model has these modules
            lora_dropout=0.01,
        )
    
    # set_grad_checkpoint(model)
    peft_model = peft.get_peft_model(model, config)
    # peft_model.train()

    # peft_model.print_trainable_parameters()

    # load the pretrained LoRA model
    if args.lora_ckpt_path is not None:
        if args.lora_ckpt_path == "huggingface-pickscore":  # download the pretrained LoRA model from huggingface
            os.makedirs('checkpoints/pretrained_lora_pickScore', exist_ok=True)
            snapshot_download(repo_id='zheyangqin/VADER_VideoCrafter_PickScore', local_dir ='checkpoints/pretrained_lora_pickScore')
            args.lora_ckpt_path = 'checkpoints/pretrained_lora_pickScore/vader_videocrafter_pickscore.pt'
        elif args.lora_ckpt_path == "huggingface-hps-aesthetic":    # download the pretrained LoRA model from huggingface
            os.makedirs('checkpoints/pretrained_lora_hps_aesthetic', exist_ok=True)
            snapshot_download(repo_id='zheyangqin/VADER_VideoCrafter_HPS_Aesthetic', local_dir ='checkpoints/pretrained_lora_hps_aesthetic')
            args.lora_ckpt_path = 'checkpoints/pretrained_lora_hps_aesthetic/vader_videocrafter_hps_aesthetic.pt'
        # load the pretrained LoRA model
        peft.set_peft_model_state_dict(peft_model, torch.load(args.lora_ckpt_path))

    # step 2.2: optimizer and loss function
    if args.use_AdamW8bit:
        optimizer = bnb.optim.AdamW8bit(peft_model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.lr)

    # rew_model = VllavaReward(accelerator.device, "/home/juntao/Models/LanguageBind/Video-LLaVA-7B", use_grad_checkpoint=True)
    cost_model = VllavaReward(accelerator.device, "/home/juntao/Projects/safe-sora/outputs/cost/checkpoint-1216", use_grad_checkpoint=True)

    # ==== load video tower ====
    if cost_model.model.config.mm_video_tower is not None:
        video_tower = cost_model.model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_tower.to(device=accelerator.device, dtype=torch.bfloat16)
    # ==== load image tower ====
    if cost_model.model.config.mm_image_tower is not None:
        image_tower = cost_model.model.get_image_tower()
        if not image_tower.is_loaded:
            image_tower.load_model()
        image_tower.to(device=accelerator.device, dtype=torch.bfloat16)

    # rew_model.load_score(
    #     score_head_path=os.path.join(args.rew_lora_ckpt_path, 'score_head.pt'),
    #     score_name="reward"
    # )
    # rew_model.load_score(
    #     score_head_path=os.path.join(args.cost_lora_ckpt_path, 'score_head.pt'),
    #     score_name="cost"
    # )

    # if accelerator.process_index in args.reward_train_processes:
    clip = CLIPReward(precision='bf16')
    hps = HPSReward(device=accelerator.device, hps_version="v2.1")
    #if accelerator.process_index in args.video_rm_train_processes:
    internv2 = InternV2Reward(rm_ckpt_dir='/home/juntao/Models/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt')

    vscore = 
    
    peft_model, optimizer = accelerator.prepare(peft_model, optimizer)
    # peft_model, optimizer, rew_model, hps = accelerator.prepare(peft_model, optimizer, rew_model, hps)

    # sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    # latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = peft_model.module.temporal_length if args.frames < 0 else args.frames
    channels = peft_model.module.channels

    ## ------------------------step 3: load data--------------------------------
    with open(args.prompt_path, "r") as f:
        prompts_cfg = json.load(f)
        prompts = [prompt_item["prompt_text"] for prompt_item in prompts_cfg]
        sources = [prompt_item["source"] for prompt_item in prompts_cfg]
        assert len(prompts) == 2919
    assert isinstance(prompts, list)

    ## -------------------step 4: run training over samples---------------------
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    first_epoch = 0
    global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, min(args.max_train_steps, args.num_train_epochs*100*args.train_batch_size)))
    progress_bar.set_description("Training Steps")
    start = time.time()

    # batch_size = args.train_batch_size
    train_dataloader = get_dataloader(prompts, sources, args.train_batch_size, distributed=True)

    cost_list = []
    train_video_rm_loss_list = []
    train_image_rm_loss_list = []

    # apply lr_scheduler
    total_steps = (args.num_train_epochs * len(train_dataloader)) / args.gradient_accumulation_steps
    logger.info(f"Total training steps = {total_steps}")
    # scheduler = cosine_lr(optimizer, args.lr, args.warmup_ratio * total_steps, total_steps)

    if accelerator.is_main_process:
        wandb.watch(peft_model, log='all', log_freq=20)

    for epoch in range(first_epoch, args.num_train_epochs):

        for batch in tqdm(train_dataloader,desc="Training Progress"):
            batch_prompts = batch['prompt']
            # batch_sources = batch['source']
            # unsafe_mask = []
            # for source in batch_sources:
            #     if source == 'safe' or source == 'dvg':
            #         unsafe_mask.append(0)
            #     elif source == 'unsafe':
            #         unsafe_mask.append(1)
            #     else:
            #         raise NotImplementedError
            # unsafe_mask = torch.tensor(unsafe_mask).to(accelerator.device)
            # assert unsafe_mask == torch.zeros(unsafe_mask.size()).to(accelerator.device)

            # randomize training process
            random.seed(datetime.datetime.now().microsecond)
            torch.manual_seed(datetime.datetime.now().microsecond)

            # Step 4.1 forward pass

            # train_prompt, promt_metadata = zip(
            #     *[prompt_fn(args.nouns_file, args.activities_file) for _ in range(args.train_batch_size)] # get train_batch_size prompts in a tuple
            #     )
            # train_prompt = list(train_prompt)  # tuple to list ['', '', '', ...]

            batch_size = len(batch_prompts)
            # assert batch_size == len(batch_prompts)
            noise_shape = [batch_size, channels, frames, h, w]

            fps = torch.tensor([args.fps]*batch_size).to(accelerator.device).long()

            # prompts = train_prompt
            # if isinstance(prompts, str):
            #     prompts = [prompts]

            with accelerator.accumulate(peft_model):    # gradient accumulation
                with accelerator.autocast():            # mixed precision
                    text_emb = peft_model.module.get_learned_conditioning(batch_prompts)

                    if args.mode == 'base':
                        cond = {"c_crossattn": [text_emb], "fps": fps}
                    else:   # TODO: implement i2v mode training in the future
                        raise NotImplementedError

                    latent_frames = 4
                    skip_frames = args.frames // latent_frames
                    start_id = torch.randint(0, skip_frames, (1,))[0].item()
                    frames_start = random.randint(0, (args.frames // skip_frames) - latent_frames)
                    decode_idxs = torch.arange(start_id, args.frames, skip_frames)[
                        frames_start : frames_start + latent_frames
                    ]
                    assert len(decode_idxs) == latent_frames

                    # prepare neg_cond
                    safety_concept = ['hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty']
                    # neg_cond = peft_model.module.get_learned_conditioning(safety_concept)
                    neg_cond = None

                    # Step 4.1: inference, batch_samples shape: batch, <samples>, c, t, h, w
                    if isinstance(peft_model, torch.nn.parallel.DistributedDataParallel):
                        batch_samples = rg_batch_ddim_sampling(peft_model.module, cond, noise_shape, args.n_samples, \
                                                            args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, \
                                                            backprop_mode=args.backprop_mode, decode_frame=args.decode_frame, \
                                                            decode_idxs=decode_idxs, vae_scale_factor=model_config["params"]["scale_factor"], neg_cond=neg_cond, **kwargs)
                    else:
                        batch_samples = rg_batch_ddim_sampling(peft_model, cond, noise_shape, args.n_samples, \
                                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, \
                                                                backprop_mode=args.backprop_mode, decode_frame=args.decode_frame, \
                                                                decode_idxs=decode_idxs, vae_scale_factor=model_config["params"]["scale_factor"], neg_cond=neg_cond, **kwargs)
                    
                    assert peft_model.module.training == True
                    # assert batch_samples.requires_grad == True

                    if args.reward_fn == 'vllava':
                        video_frames_ = batch_samples.squeeze(1)
                        val_image = video_frames_[:,:,0,:,:].clone().detach().to(torch.float16)

                        # rew_model.module.model.model.set_adapter("reward")
                        # rew_model.module.set_score("reward")
                        # vllava_reward = rew_model(video_frames_, batch_prompts)
                        # assert vllava_reward.requires_grad == True

                        # rew_model.module.model.model.set_adapter("cost")
                        # rew_model.module.set_score("cost")
                        # cost = rew_model(video_frames_, batch_prompts)
                        # assert cost.requires_grad == True

                        # rew_model.model.model.set_adapter("reward")
                        # rew_model.set_score("reward")
                        # vllava_reward = rew_model(video_frames_, batch_prompts)
                        # assert vllava_reward.requires_grad == True

                        # rew_model.model.model.set_adapter("cost")
                        # rew_model.set_score("cost")
                        # cost = rew_model(video_frames_, batch_prompts)
                        # cost = torch.zeros(vllava_reward.shape).to(accelerator.device)
                        cost = cost_model(video_frames_, batch_prompts)
                        assert cost.requires_grad == True

                        internv2_reward = torch.zeros_like(cost, requires_grad=True)
                        clip_reward = torch.zeros_like(cost, requires_grad=True)
                        hps_reward = torch.zeros_like(cost, requires_grad=True)
                        
                        # calculate video reward
                        internv2_reward = internv2(video_frames_.permute(0,2,1,3,4), batch_prompts)
                        video_rm_loss = - (0.5 * internv2_reward).mean()
                        assert video_rm_loss.requires_grad == True
                        
                        # calculate image reward
                        frame_index = random.randint(0, video_frames_.shape[2]-1)
                        clip_reward = clip(video_frames_[:,:,frame_index,:,:], batch_prompts)
                        hps_reward = hps(video_frames_[:,:,frame_index,:,:], batch_prompts)
                        # image_rm_loss = - (0.2 * clip_reward + 0.2 * hps_reward).mean()
                        image_rm_loss = - (0.2 * clip_reward).mean()
                        assert image_rm_loss.requires_grad == True
                        
                        #print(video_frames_.shape)
                        #input()
                        # frame_reward = rew_model(video_frames_[:,:,frame_index,:,:], batch_prompts)
                        # assert frame_reward.requires_grad == True
                        # print("using single frame reward")

                        # reward = (0.01 * vllava_reward) + (0.5 * internv2_reward) + (0.2 * clip_reward) + (0.2 * hps_reward)
                        # reward = (vllava_reward - REWARD_NORMALIZE["vllava_score"]["mean"]) / REWARD_NORMALIZE["vllava_score"]["std"]
                        # reward = reward + 0.1 * (internv2_reward - REWARD_NORMALIZE["internv2_score"]["mean"]) / REWARD_NORMALIZE["internv2_score"]["std"]
                        # reward = reward + 0.05 * (clip_reward - REWARD_NORMALIZE["clip_score"]["mean"]) / REWARD_NORMALIZE["clip_score"]["std"]
                        # reward = reward + 0.05 * (hps_reward - REWARD_NORMALIZE["hps_score"]["mean"]) / REWARD_NORMALIZE["hps_score"]["std"]
                        # reward = (vllava_reward + frame_reward) / 2.0
                        # reward = (0.01 * vllava_reward) + (0.2 * frame_reward)
                        # reward = vllava_reward
                        
                        cost = torch.clamp(cost, min=0.0)
                        loss = video_rm_loss + image_rm_loss + cost.mean()
                        # loss = cost.mean()
                        # loss = video_rm_loss + image_rm_loss
                        # loss = image_rm_loss +cost.mean()
                    else:
                        raise NotImplementedError

                # Gather the losses across all processes for logging (if we use distributed training).
                # loss.repeat(args.train_batch_size) is to get the total loss for each sample in the batch
                step_video_rm_loss = accelerator.gather(video_rm_loss.clone().repeat(args.train_batch_size)).mean().detach()
                train_video_rm_loss_list.append(step_video_rm_loss.item())
                step_image_rm_loss = accelerator.gather(image_rm_loss.clone().repeat(args.train_batch_size)).mean().detach()
                train_image_rm_loss_list.append(step_image_rm_loss.item())
                step_cost = accelerator.gather(cost.clone().mean().repeat(args.train_batch_size)).mean().detach()
                cost_list.append(step_cost.item())

                # Step 4.2 backpropagation
                # video_frames_.retain_grad()
                accelerator.backward(loss)

                # print(f"traced_video_grad!!!!: {traced_videos.grad[0]}")
                # print(f"NORM!!!!: {(traced_videos.grad**2).mean():.2e}")
                # print(f"traced_embeds--------: {(traced_embeds.grad**2).mean():.2e}")
                # print(f"traced_features!!!!!: {(traced_features.grad**2).mean():.2e}")
                # print(f"video_frames_!!!!!: {(video_frames_.grad**2).mean():.2e}")
                # print((video_frames_.grad**2).mean())
                # print(f"{accelerator.clip_grad_norm_(peft_model.parameters(), max_norm=float('inf')):.2e}")
                # print(accelerator.clip_grad_norm_(peft_model.parameters(), max_norm=float('inf'))<1e-8)
                if args.max_grad_norm > 0:  # gradient clipping is to prevent exploding gradients
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(peft_model.parameters(), args.max_grad_norm)

                optimizer.step()
                # if accelerator.sync_gradients:
                #     scheduler(global_step)
                optimizer.zero_grad(set_to_none=True)

                # Step 4.3 logging and save checkpoint
                if accelerator.sync_gradients:
                    global_step += 1
                    log_dict_step = {
                        "step_video_rm_loss": step_video_rm_loss.item(), 
                        "step_image_rm_loss": step_image_rm_loss.item(), 
                        "step_cost": step_cost.item()
                    }
                    accelerator.log(log_dict_step, step=global_step)
                    progress_bar.update(1)

                    avg_video_rm_loss = sum(train_video_rm_loss_list)/len(train_video_rm_loss_list)
                    avg_image_rm_loss = sum(train_image_rm_loss_list)/len(train_image_rm_loss_list)
                    avg_cost = sum(cost_list)/len(cost_list)
                    accelerator.log({"avg_video_rm_loss": avg_video_rm_loss, "avg_image_rm_loss": avg_image_rm_loss, "avg_cost": avg_cost}, step=global_step)
                    train_video_rm_loss_list = []
                    train_image_rm_loss_list = []
                    cost_list = []

                    if global_step % checkpointing_steps ==0:
                        logger.info("Saving checkpointing....")
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(peft_model)
                            # save lora model only
                            # peft_state_dict = peft_model.state_dict()
                            # peft_model_path = os.path.join(output_dir, f"model_{global_step}.pt")
                            peft_state_dict = peft.get_peft_model_state_dict(unwrapped_model)
                            peft_model_path = os.path.join(output_dir, f"peft_model_{global_step}.pt")
                            torch.save(peft_state_dict, peft_model_path)
                        logger.info("Saving checkpoing end")

                if global_step >= args.max_train_steps:
                    break

            ## --------------Validation -----------------
            ## generate videos, and compute reward and cost

            ## ---------------Step 5: Validation and save videos----------------
            if should_sample(global_step, validation_steps, args.is_sample_preview):
                ## 5.1 save the training sample
                if accelerator.is_local_main_process:
                    with torch.no_grad():
                        vis_dict = {}
                        if args.decode_frame in ["fml", "all", "alt"]:
                            ## b,samples,c,t,h,w
                            dir_name = os.path.join(output_dir, "samples", f"step_{global_step}")
                            filenames = [f"{id+1:04d}" for id in range(1)] # save only one training videl
                            # if dir_name is not exists, create it
                            os.makedirs(dir_name, exist_ok=True)
                            save_videos(batch_samples[0].unsqueeze(0), dir_name, filenames, fps=args.savefps)    # unsqueeze(0) is to add the sample dimension

                            # upload the video and their corresponding prompts to wandb
                            if args.use_wandb:
                                for i, filename in enumerate(filenames):    # len(filenames) is 1 in this case
                                    video_path = os.path.join(dir_name, f"{filename}.mp4")
                                    vis_dict[f"train_sample_{i}"] = wandb.Video(video_path, fps=args.savefps, caption=batch_prompts[i])
                        else:
                            # save as image
                            train_gen_frames = Image.fromarray((((val_image + 1.0)/2.0)[0].permute(1,2,0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))
                            gen_image = wandb.Image(train_gen_frames, caption=batch_prompts[0])
                            vis_dict["gen image (train)"] = gen_image

                    accelerator.log(vis_dict, step=global_step)
                    logger.info("Training sample saved!")

                # release the memory
                del batch_samples, video_frames_
                torch.cuda.empty_cache()
                gc.collect()

                ## Step 5.2: generate new validation videos
                with torch.no_grad():
                    vis_dict = {}
                    random.seed(args.seed)  # make sure the validation samples are the same for each epoch in order to compare the results
                    torch.manual_seed(args.seed)

                    # video validation loop\
                    for n in range(args.num_val_runs):

                        # prompts_all, promt_metadata = zip(
                        #     *[prompt_fn(args.nouns_file, args.activities_file) for _ in range(args.val_batch_size * accelerator.num_processes)] # get val_batch_size prompts in a tuple
                        #     )
                        #prompts_all = list(prompts_all)
                        prompt_idx = random.sample(range(len(prompts)), args.val_batch_size * accelerator.num_processes)

                        with accelerator.split_between_processes(prompt_idx) as val_idx:
                            # with accelerator.split_between_processes(prompts_all) as val_prompt:
                            val_prompt = [prompts[i] for i in val_idx]
                            assert len(val_prompt) == args.val_batch_size
                            # store output of generations in dict
                            results=dict(filenames=[],dir_name=[], prompt=[], gpu_no=[])

                            # Step 5.2.1: forward pass
                            val_batch_size = len(val_prompt)
                            noise_shape = [val_batch_size, channels, frames, h, w]

                            fps = torch.tensor([args.fps]*val_batch_size).to(accelerator.device).long()

                            # prompts = val_prompt
                            # if isinstance(prompts, str):
                            #     prompts = [prompts]

                            with accelerator.autocast():            # mixed precision
                                text_emb = peft_model.module.get_learned_conditioning(val_prompt).to(accelerator.device)

                                if args.mode == 'base':
                                    cond = {"c_crossattn": [text_emb], "fps": fps}
                                else:   # TODO: implement i2v mode training in the future
                                    raise NotImplementedError

                                ## Step 4.1: inference, batch_samples shape: batch, <samples>, c, t, h, w
                                # no backprop_mode=args.backprop_mode because it is inference process
                                if isinstance(peft_model, torch.nn.parallel.DistributedDataParallel):
                                    batch_samples = batch_ddim_sampling(peft_model.module, cond, noise_shape, args.n_samples, \
                                                                        args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, decode_frame=args.decode_frame, **kwargs)
                                else:
                                    batch_samples = batch_ddim_sampling(peft_model, cond, noise_shape, args.n_samples, \
                                                                            args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, decode_frame=args.decode_frame, **kwargs)

                            ## batch_samples: b,samples,c,t,h,w
                            dir_name = os.path.join(output_dir, "samples", f"step_{global_step}")
                            # filenames should be related to the gpu index
                            filenames = [f"{n}_{accelerator.local_process_index}_{id+1:04d}" for id in range(batch_samples.shape[0])] # from 0 to batch size, n is the index of the batch
                            # if dir_name is not exists, create it
                            os.makedirs(dir_name, exist_ok=True)

                            save_videos(batch_samples, dir_name, filenames, fps=args.savefps)

                            results["filenames"].extend(filenames)
                            results["dir_name"].extend([dir_name]*len(filenames))
                            results["prompt"].extend(val_prompt)
                            results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

                        # collect inference results from all the GPUs
                        results_gathered=gather_object(results)
                        # accelerator.wait_for_everyone() # wait for all processes to finish saving the videos
                        if accelerator.is_main_process:
                            filenames = []
                            dir_name = []
                            val_prompts = []
                            for i in range(len(results_gathered)):
                                filenames.extend(results_gathered[i]["filenames"])
                                dir_name.extend(results_gathered[i]["dir_name"])
                                val_prompts.extend(results_gathered[i]["prompt"])
                            # upload the video and their corresponding prompt to wandb
                            if args.use_wandb:
                                for i, filename in enumerate(filenames):
                                    video_path = os.path.join(dir_name[i], f"{filename}.mp4")
                                    vis_dict[f"{n}_sample_{i}"] = wandb.Video(video_path, fps=args.savefps, caption=val_prompts[i])

                            accelerator.log(vis_dict, step=global_step)
                            logger.info("Validation sample saved!")

                        # release the memory of validation process
                        del batch_samples
                        torch.cuda.empty_cache()
                        gc.collect()
    # save_path=os.path.join(output_dir,"model_final.pt")
    # torch.save(peft_model.state_dict(),save_path)
    logger.info("Saving checkpointing....")
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(peft_model)
        peft_state_dict = peft.get_peft_model_state_dict(unwrapped_model)
        peft_model_path = os.path.join(output_dir, f"peft_model_final.pt")
        torch.save(peft_state_dict, peft_model_path)
    logger.info("Saving checkpoing end")
    


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@VADER-VideoCrafter: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_training(args)
