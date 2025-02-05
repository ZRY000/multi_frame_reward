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
from copy import deepcopy
import json
from typing import Literal, Any

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos, print_optimizer_params
from funcs import batch_ddim_sampling, vg_batch_ddim_sampling, get_critic_optimizer, CriticDataset, compute_actor_loss, compute_critic_loss
from utils.utils import instantiate_from_config
from lvdm.models.critic import Critic, MultiFrameCritic
from lvdm.models.reward import VllavaReward
from lvdm.models.modeling_videollava import LlavaLlamaForScore, ScoreModelOutput
from lvdm.models.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
    MAX_VIDEO_LENGTH,
)

import peft
import torchvision
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers.utils import ContextManagers
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection, AutoTokenizer
from Core.aesthetic_scorer import AestheticScorerDiff
from Core.actpred_scorer import ActPredScorer
from Core.weather_scorer import WeatherScorer
from Core.compression_scorer import JpegCompressionScorer, jpeg_compressibility
import Core.prompts as prompts_file
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
import bitsandbytes as bnb
from accelerate import Accelerator, DeepSpeedPlugin
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

logger = get_logger(__name__, log_level="INFO") # get logger for current module

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return {
            'prompt': prompt,
        }
    
# Function to get sampler for distributed training
def get_sampler(dataset, distributed=False):
    if distributed:
        return DistributedSampler(dataset)
    else:
        return None

# Function to create dataloader
def get_dataloader(prompts, batch_size, distributed=False):
    dataset = PromptDataset(prompts)
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
    parser.add_argument("--critic_type", type=str, default="single-frame", help="type of critic: single-frame or multi-frame")
    parser.add_argument("--critic_ckp", type=str, default=None, help="critic checkpoint path")
    parser.add_argument("--critic_name", type=str, default="hps_clip", help="name of critic")
    parser.add_argument("--critic_ds_size", type=int, default=1, help="The size of critic dataset",)
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="weight of kl divergence")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma in MDP")
    parser.add_argument("--actor_loss_type", type=str, default="same", help="type of loss")
    parser.add_argument("--rg_type", type=str, default="draft", help="type of reward gradient baseline")
    parser.add_argument("--vg_backprop", type=int, default=2, help="Times of value gradient backpropagation",)
    parser.add_argument("--critic_method", type=str, default="td-lambda", help="name of critic")
    parser.add_argument("--lam", type=float, default=0.95, help="lambda in td-lambda")
    parser.add_argument("--critic_iterations", type=int, default=1, help="Critic training iterations",)
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for smoothing")
    parser.add_argument("--critic_lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path of critic")

    return parser


def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     torch_dtype=None):
    '''
    Args:
        aesthetic_target: float, the target value of the aesthetic score. it is 10 in this experiment
        grad_scale: float, the scale of the gradient. it is 0.1 in this experiment
        device: torch.device, the device to run the model. 
        torch_dtype: torch.dtype, the data type of the model.

    Returns:
        loss_fn: function, the loss function of the aesthetic reward function.
    '''
    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)

    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss.mean() * grad_scale, rewards.mean()
    return loss_fn


def hps_loss_fn(inference_dtype=None, device=None, hps_version="v2.0"):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of the HPS reward function.
        '''
    model_name = "ViT-H-14"
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=inference_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )    
    
    tokenizer = get_tokenizer(model_name)
    
    if hps_version == "v2.0":   # if there is a error, please download the model manually and set the path
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
    else:   # hps_version == "v2.1"
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss.mean(), scores.mean()
    
    return loss_fn

def hps_rew_fn(inference_dtype=None, device=None, hps_version="v2.0"):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of the HPS reward function.
    '''
    model_name = "ViT-H-14"

    model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=inference_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

    tokenizer = get_tokenizer(model_name)

    if hps_version == "v2.0":
        checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2_compressed.pt"
    else:   # hps_version == "v2.1"
        checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2.1_compressed.pt"

    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("hpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded\n")
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711])

    def rew_fn(im_pix, prompts):
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)

        return  scores

    return rew_fn

def aesthetic_hps_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     inference_dtype=None, 
                     device=None, 
                     hps_version="v2.0"):
    '''
    Args:
        aesthetic_target: float, the target value of the aesthetic score. it is 10 in this experiment
        grad_scale: float, the scale of the gradient. it is 0.1 in this experiment
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of a combination of aesthetic and HPS reward function.
    '''
    # HPS
    model_name = "ViT-H-14"
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=inference_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )    
    
    # tokenizer = get_tokenizer(model_name)
    
    if hps_version == "v2.0":   # if there is a error, please download the model manually and set the path
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
    else:   # hps_version == "v2.1"
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    # Aesthetic
    scorer = AestheticScorerDiff(dtype=inference_dtype).to(device, dtype=inference_dtype)
    scorer.requires_grad_(False)
    
    def loss_fn(im_pix_un, prompts):
        # Aesthetic
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)

        aesthetic_rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            aesthetic_loss = -1 * aesthetic_rewards
        else:
            # using L1 to keep on same scale
            aesthetic_loss = abs(aesthetic_rewards - aesthetic_target)
        aesthetic_loss = aesthetic_loss.mean() * grad_scale
        aesthetic_rewards = aesthetic_rewards.mean()

        # HPS
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(im_pix, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        hps_loss = abs(1.0 - scores)
        hps_loss = hps_loss.mean()
        hps_rewards = scores.mean()

        loss = (1.5 * aesthetic_loss + hps_loss) /2  # 1.5 is a hyperparameter. Set it to 1.5 because experimentally hps_loss is 1.5 times larger than aesthetic_loss
        rewards = (aesthetic_rewards + 15 * hps_rewards) / 2    # 15 is a hyperparameter. Set it to 15 because experimentally aesthetic_rewards is 15 times larger than hps_reward
        return loss, rewards
    
    return loss_fn

def pick_score_loss_fn(inference_dtype=None, device=None):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.

    Returns:
        loss_fn: function, the loss function of the PickScore reward function.
    '''
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path, torch_dtype=inference_dtype)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path, torch_dtype=inference_dtype).eval().to(device)
    model.requires_grad_(False)

    def loss_fn(im_pix_un, prompts):    # im_pix_un: b,c,h,w
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)

        # reproduce the pick_score preprocessing
        im_pix = im_pix * 255   # b,c,h,w

        if im_pix.shape[2] < im_pix.shape[3]:
            height = 224
            width = im_pix.shape[3] * height // im_pix.shape[2]    # keep the aspect ratio, so the width is w * 224/h
        else:
            width = 224
            height = im_pix.shape[2] * width // im_pix.shape[3]    # keep the aspect ratio, so the height is h * 224/w

        # interpolation and antialiasing should be the same as below
        im_pix = torchvision.transforms.Resize((height, width), 
                                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC, 
                                               antialias=True)(im_pix)
        im_pix = im_pix.permute(0, 2, 3, 1)  # b,c,h,w -> (b,h,w,c)
        # crop the center 224x224
        startx = width//2 - (224//2)
        starty = height//2 - (224//2)
        im_pix = im_pix[:, starty:starty+224, startx:startx+224, :]
        # do rescale and normalize as CLIP
        im_pix = im_pix * 0.00392156862745098   # rescale factor
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        im_pix = (im_pix - mean) / std
        im_pix = im_pix.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        
        # embed
        image_embs = model.get_image_features(pixel_values=im_pix)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        loss = abs(1.0 - scores / 100.0)
        return loss.mean(), scores.mean()
    
    return loss_fn

def weather_loss_fn(inference_dtype=None, device=None, weather="rainy", target=None, grad_scale=0):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        weather: str, the weather condition. It is "rainy" or "snowy" in this experiment.
        target: float, the target value of the weather score. It is 1.0 in this experiment.
        grad_scale: float, the scale of the gradient. It is 1 in this experiment.

    Returns:
        loss_fn: function, the loss function of the weather reward function.
    '''
    if weather == "rainy":
        reward_model_path = "../assets/rainy_reward.pt"
    elif weather == "snowy":
        reward_model_path = "../assets/snowy_reward.pt"
    else:
        raise NotImplementedError
    scorer = WeatherScorer(dtype=inference_dtype, model_path=reward_model_path).to(device, dtype=inference_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un + 1) / 2).clamp(0, 1)   # from [-1, 1] to [0, 1]
        rewards = scorer(im_pix)
        
        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)

        return loss.mean() * grad_scale, rewards.mean()
    return loss_fn

def objectDetection_loss_fn(inference_dtype=None, device=None, targetObject='dog.', model_name='grounding-dino-base'):
    '''
    This reward function is used to remove the target object from the generated video.
    We use yolo-s-tiny model to detect the target object in the generated video.

    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        targetObject: str, the object to detect. It is "dog" in this experiment.

    Returns:
        loss_fn: function, the loss function of the object detection reward function.
    '''
    if model_name == "yolos-base":
        image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base", torch_dtype=inference_dtype)
        model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base", torch_dtype=inference_dtype).to(device)
        # check if "." in the targetObject name for yolos model
        if "." in targetObject:
            raise ValueError("The targetObject name should not contain '.' for yolos-base model.")
    elif model_name == "yolos-tiny":
        image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny", torch_dtype=inference_dtype)
        model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", torch_dtype=inference_dtype).to(device)
        # check if "." in the targetObject name for yolos model
        if "." in targetObject:
            raise ValueError("The targetObject name should not contain '.' for yolos-tiny model.")
    elif model_name == "grounding-dino-base":
        image_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", torch_dtype=inference_dtype)
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base",torch_dtype=inference_dtype).to(device)
        # check if "." in the targetObject name for grounding-dino model
        if "." not in targetObject:
            raise ValueError("The targetObject name should contain '.' for grounding-dino-base model.")
    elif model_name == "grounding-dino-tiny":
        image_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", torch_dtype=inference_dtype)
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny", torch_dtype=inference_dtype).to(device)
        # check if "." in the targetObject name for grounding-dino model
        if "." not in targetObject:
            raise ValueError("The targetObject name should contain '.' for grounding-dino-tiny model.")
    else:
        raise NotImplementedError
    
    model.requires_grad_(False)
    model.eval()

    def loss_fn(im_pix_un): # im_pix_un: b,c,h,w
        images = ((im_pix_un / 2) + 0.5).clamp(0.0, 1.0)

        # reproduce the yolo preprocessing
        height = 512
        width = 512 * images.shape[3] // images.shape[2]    # keep the aspect ratio, so the width is 512 * w/h
        images = torchvision.transforms.Resize((height, width), antialias=False)(images)
        images = images.permute(0, 2, 3, 1)  # b,c,h,w -> (b,h,w,c)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        image_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        images = (images - image_mean) / image_std
        normalized_image = images.permute(0,3,1,2)  # NHWC -> NCHW

        # Process images
        if model_name == "yolos-base" or model_name == "yolos-tiny":
            outputs = model(pixel_values=normalized_image)
        else:   # grounding-dino model
            inputs = image_processor(text=targetObject, return_tensors="pt").to(device)
            outputs = model(pixel_values=normalized_image, input_ids=inputs.input_ids)
        
        # Get target sizes for each image
        target_sizes = torch.tensor([normalized_image[0].shape[1:]]*normalized_image.shape[0]).to(device)

        # Post-process results for each image
        if model_name == "yolos-base" or model_name == "yolos-tiny":
            results = image_processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=target_sizes)
        else:   # grounding-dino model
            results = image_processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=0.4,
                        text_threshold=0.3,
                        target_sizes=target_sizes
                    )

        sum_avg_scores = 0
        for i, result in enumerate(results):
            if model_name == "yolos-base" or model_name == "yolos-tiny":
                id = model.config.label2id[targetObject]
                # get index of targetObject's label
                index = torch.where(result["labels"] == id) 
                if len(index[0]) == 0:  # index: ([],[]) so index[0] is the first list
                    sum_avg_scores = torch.sum(outputs.logits - outputs.logits)    # set sum_avg_scores to 0
                    continue
                scores = result["scores"][index]
            else:   # grounding-dino model
                if result["scores"].shape[0] == 0:
                    sum_avg_scores = torch.sum(outputs.last_hidden_state - outputs.last_hidden_state)   # set sum_avg_scores to 0
                    continue
                scores = result["scores"]
            sum_avg_scores = sum_avg_scores +  (torch.sum(scores) / scores.shape[0])

        loss = sum_avg_scores / len(results)
        reward = 1 - loss

        return loss, reward
    return loss_fn

def compression_loss_fn(inference_dtype=None, device=None, target=None, grad_scale=0, model_path=None):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        model_path: str, the path of the compression model.

    Returns:
        loss_fn: function, the loss function of the compression reward function.
    '''
    scorer = JpegCompressionScorer(dtype=inference_dtype, model_path=model_path).to(device, dtype=inference_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un + 1) / 2).clamp(0, 1)
        rewards = scorer(im_pix)
        
        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)
        return loss.mean() * grad_scale, rewards.mean()
    
    return loss_fn

def actpred_loss_fn(inference_dtype=None, device=None, num_frames = 14, target_size=224):
    scorer = ActPredScorer(device=device, num_frames = num_frames, dtype=inference_dtype)
    scorer.requires_grad_(False)

    def preprocess_img(img):
        img = ((img/2) + 0.5).clamp(0,1)
        img = torchvision.transforms.Resize((target_size, target_size), antialias = True)(img)
        img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img
    def loss_fn(vid, target_action_label):
        vid = torch.cat([preprocess_img(img).unsqueeze(0) for img in vid])[None]
        return scorer.get_loss_and_score(vid, target_action_label)
    
    return loss_fn


def should_sample(global_step, validation_steps, is_sample_preview):
    return (global_step % validation_steps == 0 or global_step ==1)  \
    and is_sample_preview

EVAL_TEMPLATE_FILE = '/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/scripts/ds_zero2.json'
def get_deepspeed_eval_config(
    *,
    stage: int = 3,
    offload: Literal['none', 'parameter', 'optimizer', 'all'] = 'none',
    fp16: bool = False,
    bf16: bool = False,
) -> dict[str, Any]:
    """Get the DeepSpeed config for evaluation.

    Args:
        stage (int, optional): The stage of ZeRO. Defaults to 3.
        offload (Literal['none', 'parameter', 'optimizer', 'all'], optional): The offload mode.
        fp16 (bool, optional): Whether to use FP16 precision. Defaults to False.
        bf16 (bool, optional): Whether to use BF16 precision. Defaults to False.

    Returns:
        The DeepSpeed config for evaluation.
    """
    assert offload in {'none', 'parameter', 'optimizer', 'all'}

    with open(EVAL_TEMPLATE_FILE, mode='rt', encoding='utf-8') as f:
        eval_config = json.load(f)

    if stage in {1, 2}:
        # The evaluation config only works for ZeRO stage 0 and ZeRO stage 3
        stage = 0

    eval_config['train_batch_size'] = None
    eval_config['train_micro_batch_size_per_gpu'] = 1
    eval_config['gradient_accumulation_steps'] = 1
    eval_config['zero_optimization']['stage'] = stage
    if offload in {'parameter', 'all'}:
        eval_config['zero_optimization'].setdefault('offload_param', {})
        eval_config['zero_optimization']['offload_param']['device'] = 'cpu'
    if fp16 or 'fp16' in eval_config:
        eval_config.setdefault('fp16', {})
        eval_config['fp16']['enabled'] = fp16
    if bf16 or 'bf16' in eval_config:
        eval_config.setdefault('bf16', {})
        eval_config['bf16']['enabled'] = bf16
    return eval_config

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
            exp_name = f"vc_critic_pretrain_{args.backprop_mode}"
            wandb_args['name'] = f"{exp_name}-{current_time}"
            
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

    # convert first_stage_model and cond_stage_model to torch.float16 if mixed_precision is True
    if args.mixed_precision != 'no':
        model.first_stage_model = model.first_stage_model.half()
        model.cond_stage_model = model.cond_stage_model.half()

    # step 2.1: add LoRA using peft
    config = peft.LoraConfig(
            r=args.lora_rank,
            target_modules=["to_k", "to_v", "to_q"],        # only diffusion_model has these modules
            lora_dropout=0.01,
        )
    
    peft_model = peft.get_peft_model(model, config)

    peft_model.print_trainable_parameters()

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
    
    # build critic model
    # critic is only used for optimization
    # all the other computation is done using target_critic

    # assert args.critic_type == "single-frame"
    # critic = Critic(args, peft_model.dtype, accelerator.device, ckp_path=args.critic_ckp)
    # logger.info(f"Using Single-Frame Critic!")

    # video-llava backbone
    rew_model = VllavaReward(accelerator.device, "/home/juntao/Projects/safe-sora/outputs/harm7/reward-harmlessness", use_grad_checkpoint=True)
    
    if rew_model.model.config.mm_video_tower is not None:
        video_tower = rew_model.model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_tower.to(device=accelerator.device, dtype=torch.bfloat16)
    
    rew_model.eval()
    # ds_config = '/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/scripts/ds_zero2.json'
    # rew_model_engine, _, _, _ = deepspeed.initialize(
    #     model=rew_model,
    #     config=ds_config,
    #     model_parameters=None  
    # )
    
    if args.critic_type == "single-frame":
        critic = Critic(args, peft_model.dtype, accelerator.device, ckp_path=args.critic_ckp)
        logger.info(f"Using Single-Frame Critic!")
    elif args.critic_type == "multi-frame" and args.critic_name == "languagebind_video":
        critic = MultiFrameCritic(peft_model.dtype, accelerator.device, ckp_path=args.critic_ckp)
        if args.critic_lora_enabled:
            critic.model.convert_to_lora()
        logger.info(f"Using Multi-Frame Critic!")
    elif args.critic_type == "multi-frame" and args.critic_name == "video_llava":
        # critic = VllavaReward(accelerator.device, "/home/juntao/Projects/safe-sora/outputs/vc/reward-helpfulness", use_grad_checkpoint=True)
        critic_config = peft.LoraConfig(
            r=64,
            lora_alpha=16,  
            target_modules=find_all_linear_names(rew_model), 
            lora_dropout=0.05,  
            bias='none',
        )
        critic = peft.get_peft_model(rew_model, critic_config)
        critic.print_trainable_parameters()
        if args.critic_lora_ckpt_path is not None:
            peft.set_peft_model_state_dict(critic, torch.load(args.critic_lora_ckpt_path))
        
        if critic.model.model.config.mm_video_tower is not None:
            video_tower = critic.model.model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=accelerator.device, dtype=torch.bfloat16)
    else:
        raise NotImplementedError(f"Critic type {args.critic_type} is not supported for value gradient.")

    target_critic = deepcopy(critic)
    target_critic.eval()

    # step 2.2: optimizer and loss function
    critic_optimizer = get_critic_optimizer(args, critic)
    
    # step 2.3: learning rate scheduler
    # TODO: implement learning rate scheduler if needed
    if args.reward_fn == "aesthetic":
        loss_fn = aesthetic_loss_fn(grad_scale=0.1,
                                    aesthetic_target=10,
                                    torch_dtype = peft_model.dtype,
                                    device = accelerator.device)
    elif args.reward_fn == "hps":
        loss_fn = hps_loss_fn(peft_model.model.first_stage_model.dtype, accelerator.device, hps_version=args.hps_version)
    elif args.reward_fn == "aesthetic_hps":
        # if use mixed precision, the dtype of peft_model is torch.float32
        # while the dtype of peft_model.model.first_stage_model is torch.float16
        loss_fn = aesthetic_hps_loss_fn(aesthetic_target=10,
                                        grad_scale=0.1,
                                        inference_dtype=peft_model.dtype,
                                        device=accelerator.device,
                                        hps_version=args.hps_version)
    elif args.reward_fn == "pick_score":
        loss_fn = pick_score_loss_fn(peft_model.dtype, accelerator.device)
    elif args.reward_fn == "rainy":
        loss_fn = weather_loss_fn(peft_model.dtype, accelerator.device, weather="rainy", target=1, grad_scale=1)
    elif args.reward_fn == "snowy":
        loss_fn = weather_loss_fn(peft_model.dtype, accelerator.device, weather="snowy", target=1, grad_scale=1)
    elif args.reward_fn == "objectDetection":
        loss_fn = objectDetection_loss_fn(peft_model.dtype, accelerator.device, targetObject=args.target_object, model_name=args.detector_model)
    elif args.reward_fn == "compression_score":
        loss_fn = compression_loss_fn(peft_model.dtype, accelerator.device, target=None, grad_scale=1.0, model_path=args.compression_model_path)
    elif args.reward_fn == "actpred":
        assert args.decode_frame in ['alt', 'last'], "decode_frame must be 'alt' or 'last' for actpred reward function."
        num_score_frames = peft_model.temporal_length if args.frames < 0 else args.frames
        num_score_frames = num_score_frames//2 if args.decode_frame == 'alt' else num_score_frames
        loss_fn = actpred_loss_fn(peft_model.model.first_stage_model.dtype, accelerator.device, num_frames = num_score_frames )

    # rew_fn = hps_rew_fn(peft_model.model.first_stage_model.dtype, accelerator.device, hps_version=args.hps_version)
    # load reward model for computing rew_fn
    '''
    rew_model = VllavaReward(accelerator.device, "/home/juntao/Projects/safe-sora/outputs/vc/reward-helpfulness")
    
    if rew_model.model.config.mm_video_tower is not None:
        video_tower = rew_model.model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_tower.to(device=accelerator.device, dtype=torch.bfloat16)
    
    rew_model.eval()
    ds_config = '/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/scripts/ds_zero2.json'
    rew_model_engine, _, _, _ = deepspeed.initialize(
        model=rew_model,
        config=ds_config,
        model_parameters=None  
    )
    '''

    peft_model, critic, target_critic, critic_optimizer = accelerator.prepare(
        peft_model, critic, target_critic, critic_optimizer)

    # sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    # latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = peft_model.module.temporal_length if args.frames < 0 else args.frames
    channels = peft_model.module.channels


    ## ------------------------step 3: load data--------------------------------
    with open(args.prompt_path, "r") as f:
        # prompts = [line.strip() for line in f.readlines()]
        prompts = json.load(f)
    assert isinstance(prompts, list)

    ## -------------------step 4: run 
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch training over samples---------------------# Train!size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    first_epoch = 0
    global_step = 0
    actor_iter_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, min(args.max_train_steps, args.num_train_epochs*100*args.train_batch_size)))
    progress_bar.set_description("Training Steps")
    start = time.time()

    # batch_size = args.train_batch_size
    train_dataloader = get_dataloader(prompts, args.train_batch_size, distributed=True)

    train_loss_list = []
    reward_list = []

    critic_dataset = CriticDataset(max_size=args.critic_ds_size)


    for epoch in range(first_epoch, args.num_train_epochs):

        for batch in tqdm(train_dataloader,desc="Training Progress"):
            batch_prompts = batch['prompt']
        # for idx, step in enumerate(range(default_iterations)): # default number of iterations is 100

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
            
            with accelerator.accumulate(critic):    # gradient accumulation
                with accelerator.autocast():            # mixed precision
                    text_emb = peft_model.module.get_learned_conditioning(batch_prompts)
                    if args.mode == 'base':
                        cond = {"c_crossattn": [text_emb], "fps": fps}
                    else:   # TODO: implement i2v mode training in the future
                        raise NotImplementedError

                    # prepare for rollouts
                    start_idx = 0
                    whole_decode = []
                    z = None
                    rew_buf = torch.zeros(args.ddim_steps, batch_size, \
                                          dtype = peft_model.module.first_stage_model.dtype, device = accelerator.device)

                    if args.critic_type == "single-frame":
                        latent_frames = 1
                    elif args.critic_type == "multi-frame":
                        latent_frames = 4
                        logger.info(f"decoding latnet frames: {latent_frames}")
                    else:
                        raise NotImplementedError(f"Critic type {args.critic_type} invalid.")

                    # idx_start = random.randint(0,noise_shape[2]-latent_frames)
                    # decode_idxs = range(idx_start, idx_start+latent_frames)
                    skip_frames = args.frames // latent_frames
                    start_id = torch.randint(0, skip_frames, (1,))[0].item()
                    frames_start = random.randint(0, (args.frames // skip_frames) - latent_frames)
                    decode_idxs = torch.arange(start_id, args.frames, skip_frames)[
                        frames_start : frames_start + latent_frames
                    ]
                    assert len(decode_idxs) == latent_frames

                    if args.rg_type == "draft":
                        final_step = args.ddim_steps
                    elif args.rg_type == "refl":
                        final_step = random.randint(args.ddim_steps//3,(args.ddim_steps//3)*2)
                    else:
                        raise NotImplementedError("RG type not supported")
                    
                    backprop_idxs = random.sample(range(1,final_step), args.vg_backprop) + [0, final_step]
                    backprop_idxs = sorted(backprop_idxs)

                    # value backpropagation
                    is_last = False
                    step_loss_list = []
                    for i in range(len(backprop_idxs)-1):
                        start_idx = backprop_idxs[i]
                        end_idx = backprop_idxs[i+1]
                        if end_idx == final_step: is_last=True
                        if isinstance(peft_model, torch.nn.parallel.DistributedDataParallel):
                            z, z_decode, traj_decode = vg_batch_ddim_sampling(peft_model.module, cond, noise_shape, args.n_samples, \
                                                                        args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, \
                                                                        backprop_mode=args.backprop_mode, z=z, start_idx=start_idx, end_idx=end_idx, \
                                                                        is_last=is_last, decode_idxs=decode_idxs, rg_type=args.rg_type, \
                                                                        vae_scale_factor = model_config["params"]["scale_factor"], **kwargs)
                        else:
                            z, z_decode, traj_decode = vg_batch_ddim_sampling(peft_model, cond, noise_shape, args.n_samples, \
                                                                        args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, \
                                                                        backprop_mode=args.backprop_mode, z=z, start_idx=start_idx, end_idx=end_idx, \
                                                                        is_last=is_last, decode_idxs=decode_idxs, rg_type=args.rg_type, \
                                                                        vae_scale_factor = model_config["params"]["scale_factor"], **kwargs)
                        # assert z_decode.requires_grad == True
                        z_decode.detach()
                        assert traj_decode.requires_grad == False
                        frame_index = random.randint(0,z_decode.shape[2]-1)         # z_decode: b,c,f,h,w; [-1,1]
                        val_img = z_decode[:,:,frame_index,:,:].clone().detach()

                        if is_last:
                            if args.reward_fn == "aesthetic" or args.reward_fn == "objectDetection" or args.reward_fn == "compression_score":
                                reward = rew_fn(z_decode[:,:,frame_index,:,:])

                            elif args.reward_fn == 'vllava':
                                with torch.no_grad():
                                    reward = rew_model(z_decode, batch_prompts)
                                assert reward.requires_grad == False

                            else:   # 'hps' or 'pick_score' or 'aesthetic_hps'
                                reward = rew_fn(z_decode[:,:,frame_index,:,:], batch_prompts)
                        else: reward = 0
                        
                        # set reward buffer for updating target values 
                        if is_last: 
                            rew_buf[-1] = reward.clone().detach()
                            step_reward = accelerator.gather(rew_buf[-1].mean().repeat(args.train_batch_size)).mean().item()
                            reward_list.append(step_reward)
                        
                        if args.critic_type == "single-frame":
                            z_decode = z_decode.squeeze(2)
                            traj_decode = traj_decode.squeeze(3)
                        
                        traj_len = traj_decode.shape[0]
                        assert traj_len == end_idx - start_idx
                        whole_decode.append(traj_decode.detach())
                        # if not (args.actor_loss_type == "distill" and is_last):
                        actor_loss = compute_actor_loss(
                            target_critic=target_critic,
                            reward=reward,
                            end_idx=end_idx,
                            traj_len=traj_len,
                            z_decode=z_decode,
                            prompts=batch_prompts,
                            gamma=args.gamma,
                            loss_type=args.actor_loss_type,
                            is_last=is_last
                        )

                        # delete z_decode after actor_loss
                        if is_last: sample_result = z_decode.clone().detach()

                        gathered_loss = accelerator.gather(actor_loss.repeat(args.train_batch_size)).mean()
                        step_loss_list.append(gathered_loss.clone().detach().item())
                        train_loss_list.append(gathered_loss.clone().detach().item())

                        actor_iter_step += 1
                        logger.info(f"Actor iteration steps: {actor_iter_step}, global_step: {global_step}, start_idx: {start_idx}, end_idx: {end_idx}")

                        # free memory
                        del z_decode, traj_decode, actor_loss, gathered_loss
                    
                    # free memory
                    del z, reward

                    step_loss = sum(step_loss_list)/len(step_loss_list)
                    step_loss_list = []

                    # update the critic using the whole trajectory
                    logger.info("Sampling process ends, start training critic")
                    whole_decode = torch.cat(whole_decode)

                    # z_decode, traj_decode -> target_values
                    # collect previous trajs and do off-policy learning
                    with torch.no_grad():
                        critic_dataset.update_traj(traj=whole_decode, end_state=sample_result, rew_buf=rew_buf, is_last=True) # TODO: correct it when dataset_size > 1 DONE
                        
                    if len(critic_dataset) < critic_dataset.max_size:
                        logger.info(f" Critic dataset current size: {len(critic_dataset)}, actor iteration steps: {actor_iter_step}")

                    # TODO: big modification
                    with torch.no_grad():
                        critic_dataset.update_target_values(
                            target_critic=target_critic,
                            prompts=batch_prompts,
                            critic_method=args.critic_method,
                            gamma=args.gamma,
                            lam=args.lam
                        )

                    critic_iterations = args.critic_iterations

                    # fit the critic
                    for i in range(critic_iterations):
                        critic_dataset.shuffle()
                        # TODO: add avg_critic_loss
                        critic_loss_list = []
                        for j in range(len(critic_dataset)):
                            critic_loss = compute_critic_loss(critic=critic,
                                                              traj=critic_dataset.traj_ds[j],
                                                              target_values=critic_dataset.target_values_ds[j],
                                                              prompts=batch_prompts)

                            accelerator.backward(critic_loss)

                            gathered_critic_loss = accelerator.gather(critic_loss.repeat(args.train_batch_size)).mean()
                            critic_loss_list.append(gathered_critic_loss.detach().item())
                            
                            # TODO: add more process
                            if args.max_grad_norm > 0:  # gradient clipping is to prevent exploding gradients
                                if accelerator.sync_gradients:
                                    accelerator.clip_grad_norm_(critic.module.parameters(), args.max_grad_norm)

                            critic_optimizer.step()
                            critic_optimizer.zero_grad(set_to_none=True)

                    # calculate average critic loss
                    avg_critic_loss = sum(critic_loss_list)/len(critic_loss_list)
                    
                    # update target critic
                    if accelerator.sync_gradients:
                        with torch.no_grad():
                            alpha = args.alpha
                            for param, param_targ in zip(critic.module.parameters(), target_critic.module.parameters()):
                                param_targ.data.mul_(alpha)
                                param_targ.data.add_((1. - alpha) * param.data)
                    
                    with torch.no_grad():
                        last_step_value = target_critic(whole_decode[-1], batch_prompts).mean().detach().item()

                    # free memory
                    del critic_loss, whole_decode

                
            # Step 4.3 logging and save checkpoint
            if accelerator.sync_gradients:
                global_step += 1
                log_dict_step = {"step_loss": step_loss,
                                    "rank0_reward": rew_buf[-1].mean().item(),
                                    "step_reward": step_reward, 
                                    "rank0_last_step_value": last_step_value}
                accelerator.log(log_dict_step, step=global_step)

                for log_key in log_dict_step:
                    logger.info(f"{log_key}: {log_dict_step[log_key]}")

                avg_loss = sum(train_loss_list)/len(train_loss_list)
                avg_reward = sum(reward_list)/len(reward_list)
                log_avg_dict = {"avg_loss": avg_loss, 
                                "avg_reward": avg_reward, 
                                "avg_critic_loss": avg_critic_loss}
                accelerator.log(log_avg_dict, step=global_step)

                for log_key in log_avg_dict:
                    logger.info(f"{log_key}: {log_avg_dict[log_key]}")

                train_loss_list = []
                reward_list = []

                progress_bar.update(1)
                torch.cuda.empty_cache()
                        

        # save critic model
        logger.info(f"Saving checkpointing for epoch {epoch}")
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            unwrapped_critic_model = accelerator.unwrap_model(target_critic)
            critic_state_dict = unwrapped_critic_model.state_dict()
            
            critic_model_dir = os.path.join(output_dir, "critic_model")
            os.makedirs(critic_model_dir, exist_ok=True)
            critic_model_path = os.path.join(critic_model_dir, f"critic_epoch_{epoch}.pt")

            torch.save(critic_state_dict, critic_model_path)
                            
            logger.info("Saving checkpoing end")


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@critic_pretrain: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_training(args)