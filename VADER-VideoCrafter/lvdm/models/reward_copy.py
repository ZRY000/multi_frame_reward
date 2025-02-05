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
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
from dataclasses import dataclass
from copy import deepcopy
from typing import List

import peft
import torchvision
from transformers.utils import ContextManagers
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection, AutoTokenizer
from transformers.modeling_utils import ModelOutput
from Core.aesthetic_scorer import AestheticScorerDiff
from Core.actpred_scorer import ActPredScorer
from Core.weather_scorer import WeatherScorer
from Core.compression_scorer import JpegCompressionScorer, jpeg_compressibility
import Core.prompts as prompts_file
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
from lvdm.models.modeling_videollava import LlavaLlamaForScore, ScoreModelOutput
from lvdm.models.idefics2 import Idefics2ForSequenceClassification
from utils.video_utils import video_process, image_process
from utils.checkpoint import set_grad_checkpoint
from utils.reward_utils import preprocess_multimodal, preprocess_text
from lvdm.models.intern_vid2.demo_config import Config, eval_dict_leaf
from lvdm.models.intern_vid2.demo_utils import setup_internvideo2
from torchvision.transforms import (
    Normalize,
    Resize,
    InterpolationMode,
    CenterCrop,
    RandomCrop,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomHorizontalFlipVideo,
)
import torchvision.transforms.functional as F
from torchvision.transforms import Compose
CLIP_RESIZE = Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
CLIP_NORMALIZE = Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)
ViCLIP_NORMALIZE = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
import open_clip

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

from lvdm.models.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
    MAX_VIDEO_LENGTH,
)

def vllava_rew_fn(model, tokenizer, device: str = 'cuda', **kwargs,):
    # kwargs['device_map'] = {'': device}
    # kwargs['torch_dtype'] = torch.float16
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # model = LlavaLlamaForScore.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    # ==========================================================================================================
    # processor = {'image': None, 'video': None}

    # mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    # mm_use_im_patch_token = getattr(model.config, 'mm_use_im_patch_token', True)
    # if mm_use_im_patch_token:
    #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #     tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    #     tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))

    # if model.config.mm_image_tower is not None:
    #     image_tower = model.get_image_tower()
    #     if not image_tower.is_loaded:
    #         image_tower.load_model()
    #     image_tower.to(device=device, dtype=torch.float16)
    #     image_processor = image_tower.image_processor
    #     processor['image'] = image_processor

    # if model.config.mm_video_tower is not None:
    #     video_tower = model.get_video_tower()
    #     if not video_tower.is_loaded:
    #         video_tower.load_model()
    #     video_tower.to(device=device, dtype=torch.float16)
    #     video_processor = video_tower.video_processor
    #     processor['video'] = video_processor

    # model = model.to(device)

    def rew_fn(videos, prompts):
        # prepare inputs
        input_ids = tokenizer(
            prompts,
            return_tensors='pt',
            padding='longest',
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        input_ids = input_ids[:, : tokenizer.model_max_length]
        
        videos = video_process(videos)
        new_videos = []
        for i in range(videos.shape[0]):
            new_videos.append(videos[i])
        # for image in images:
        #     if isinstance(image, list):
        #         for i in image:
        #             new_images.append(i)
        #     else:
        #         new_images.append(image)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        
        batch = {
            'input_ids': input_ids.to(model.device),
            'images': [video.to(model.device) for video in new_videos],
            'attention_mask': attention_mask.to(model.device),
        }
        outputs: ScoreModelOutput = model(**batch)
        end_scores = outputs.end_scores
        return end_scores

    return rew_fn

class VllavaReward(nn.Module):
    def __init__(self, device, ckpt_path, use_grad_checkpoint=False, **model_kwargs):
        super().__init__()
        model_kwargs['device_map'] = {'': device}
        model_kwargs['torch_dtype'] = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = LlavaLlamaForScore.from_pretrained(ckpt_path, low_cpu_mem_usage=True, **model_kwargs)

        mm_use_im_start_end = getattr(self.model.config, 'mm_use_im_start_end', False)
        mm_use_im_patch_token = getattr(self.model.config, 'mm_use_im_patch_token', True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # if self.model.config.mm_video_tower is not None:
        #     video_tower = self.model.get_video_tower()
        #     if not video_tower.is_loaded:
        #         video_tower.load_model()
        #     video_tower.to(device=device, dtype=torch.float16)
            # video_processor = video_tower.video_processor
            # processor['video'] = video_processor
        self.model = self.model.to(device)
        self.use_grad_checkpoint = use_grad_checkpoint
        if self.use_grad_checkpoint:
            set_grad_checkpoint(self.model)
            self.model.enable_input_require_grads()

        self.score_dict = {}
    
    def forward(self, videos, prompts):
        input_ids = []
        for prompt in prompts:
            if len(videos.shape) == 5:
                conversation = [
                    {
                        'from': 'human',
                        'value': f"##Video Generation Prompt: {prompt}",
                    },
                    {'from': 'gpt', 'value': '##Generated Video: \n<video>'},
                ]
            elif len(videos.shape) == 4:
                conversation = [
                    {
                        'from': 'human',
                        'value': f"##Video Generation Prompt: {prompt}",
                    },
                    {'from': 'gpt', 'value': '##Generated Video: \n<image>'},
                ]
            else:
                raise NotImplementedError

            sources = preprocess_multimodal(
                deepcopy([conversation]),
                num_frames=8,
                mm_use_im_start_end=self.model.config.mm_use_im_start_end,
            )
            data_dict = preprocess_text(sources, self.tokenizer, has_image=True)
            input_ids.append(data_dict['input_ids'][0])
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        
        if len(videos.shape) == 5:
            videos = video_process(videos)
        elif len(videos.shape) == 4:
            videos = image_process(videos)
        
        new_videos = []
        for i in range(videos.shape[0]):
            new_videos.append(videos[i])
        
        # for image in images:
        #     if isinstance(image, list):
        #         for i in image:
        #             new_images.append(i)
        #     else:
        #         new_images.append(image)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        batch = {
            'input_ids': input_ids.to(self.model.device),
            'images': [video.to(self.model.device) for video in new_videos],
            'attention_mask': attention_mask.to(self.model.device),
        }
        if self.use_grad_checkpoint: 
            outputs: ScoreModelOutput = checkpoint.checkpoint(self.model, **batch, use_reentrant=False)

        else:
            outputs: ScoreModelOutput = self.model(**batch)

        return outputs.end_scores.squeeze(-1)
    
    def load_score(self, score_head_path, score_name):
        score_head = deepcopy(self.model.score)
        score_head.load_state_dict(torch.load(score_head_path))
        self.score_dict[score_name] = score_head

    def set_score(self, score_name):
        self.model.score = self.score_dict[score_name]

class ResizeCropMinSize(nn.Module):

    def __init__(self, min_size, interpolation=InterpolationMode.BICUBIC, fill=0):
        super().__init__()
        if not isinstance(min_size, int):
            raise TypeError(f"Size should be int. Got {type(min_size)}")
        self.min_size = min_size
        self.interpolation = interpolation
        self.fill = fill
        self.random_crop = RandomCrop((min_size, min_size))

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.min_size / float(min(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            img = self.random_crop(img)
        return img

class InternV2Reward(nn.Module):
    def __init__(self, rm_ckpt_dir: str, precision="amp", n_frames=4):
        super().__init__()
        # config = Config.from_file("intern_vid2/configs/internvideo2_stage2_config.py")
        config = Config.from_file("/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/lvdm/models/intern_vid2/configs/internvideo2_stage2_config.py")
        config = eval_dict_leaf(config)
        config["inputs"]["video_input"]["num_frames"] = n_frames
        config["inputs"]["video_input"]["num_frames_test"] = n_frames
        config["model"]["vision_encoder"]["num_frames"] = n_frames

        config["model"]["vision_encoder"]["pretrained"] = rm_ckpt_dir
        config["pretrained_path"] = rm_ckpt_dir

        self.vi_clip, self.tokenizer = setup_internvideo2(config)
        self.vi_clip.requires_grad_(True)
        if precision == "fp16":
            self.vi_clip.to(torch.float16)

        # self.viclip_resize = ResizeCropMinSize(224)
        self.transfer = Compose(
            [
                CenterCropVideo((224, 224)),
            ],
        )

    def forward(self, image_inputs: torch.Tensor, text_inputs: str):
        # Process pixels and multicrop
        ori_shape = image_inputs.shape
        image_inputs = torch.stack([self.transfer(image_inputs[i]) for i in range(image_inputs.shape[0])])
        #assert image_inputs.shape[:3] == ori_shape[:3]
        device = image_inputs.device
        self.vi_clip.to(device)
        b, t = image_inputs.shape[:2]
        # image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        image_inputs = image_inputs.reshape(b * t, *image_inputs.shape[2:])
        # pixel_values = ViCLIP_NORMALIZE(self.viclip_resize(image_inputs))
        pixel_values = ViCLIP_NORMALIZE(image_inputs)
        # pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        pixel_values = pixel_values.reshape(b, t, *pixel_values.shape[1:])
        video_features = self.vi_clip.get_vid_feat_with_grad(pixel_values)

        with torch.no_grad():
            text = self.tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            ).to(device)
            _, text_features = self.vi_clip.encode_text(text)
            text_features = self.vi_clip.text_proj(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (video_features * text_features).sum(-1)
        return score

class CLIPReward(nn.Module):
    def __init__(self, precision="amp"):
        super().__init__()
        assert precision in ["bf16", "fp16", "amp", "fp32"]
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=precision,
            device="cuda",
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            pretrained_image=False,
            output_dict=True,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.model.to(torch.bfloat16)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        # Process pixels and multicrop
        self.model.to(image_inputs.device)
        image_inputs = CLIP_RESIZE(image_inputs)
        image_inputs = CLIP_NORMALIZE(image_inputs)

        if isinstance(text_inputs[0], str):
            text_inputs = self.tokenizer(text_inputs).to(image_inputs.device)

        # embed
        image_features = self.model.encode_image(image_inputs, normalize=True)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs, normalize=True)

        clip_score = (image_features * text_features).sum(-1)
        if return_logits:
            clip_score = clip_score * self.model.logit_scale.exp()
        return clip_score

class HPSReward(nn.Module):
    def __init__(self, precision='bf16', device=None, hps_version="v2.1", use_grad_checkpoint=False):
        super().__init__()
        model_name = "ViT-H-14"
    
        self.model, preprocess_train, preprocess_val = create_model_and_transforms(
                model_name,
                'laion2B-s32B-b79K',
                precision=precision,
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
        
        self.tokenizer = get_tokenizer(model_name)
        
        if hps_version == "v2.0":   # if there is a error, please download the model manually and set the path
            checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2_compressed.pt"
        else:   # hps_version == "v2.1"
            checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2.1_compressed.pt"
        # force download of model via score
        # hpsv2.score([], "", hps_version=hps_version)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model = self.model.to(device, dtype=torch.bfloat16)
        self.model.eval()

        self.target_size =  (224, 224)
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        self.device = device
        self.use_grad_checkpoint = use_grad_checkpoint
        if self.use_grad_checkpoint:
            set_grad_checkpoint(self.model)
            self.model.enable_input_require_grads()

    def forward(self, im_pix, prompts):
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(self.target_size)(im_pix)
        x_var = self.normalize(x_var).to(im_pix.dtype)        
        caption = self.tokenizer(prompts)
        caption = caption.to(self.device)
        if self.use_grad_checkpoint: 
            outputs = checkpoint.checkpoint(self.model, x_var, caption, use_reentrant=False)
        else:
            outputs = self.model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)

        return scores
    
class VideoScore(nn.Module):
    def __init__(self, device, ckpt_path, use_grad_checkpoint=False):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(ckpt_path,torch_dtype=torch.bfloat16)
        self.model = Idefics2ForSequenceClassification.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16
        ).eval()
        self.device = device
        self.model.to(self.device)

        REGRESSION_QUERY_PROMPT = """
        Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
        please watch the following frames of a given video and see the text prompt for generating the video,
        then give scores from 5 different dimensions:
        (1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
        (2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
        (3) dynamic degree, the degree of dynamic changes
        (4) text-to-video alignment, the alignment between the text prompt and the video content
        (5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

        for each dimension, output a float number from 1.0 to 4.0,
        the higher the number is, the better the video performs in that sub-score, 
        the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
        Here is an output example:
        visual quality: 3.2
        temporal consistency: 2.7
        dynamic degree: 4.0
        text-to-video alignment: 2.3
        factual consistency: 1.8

        For this video, the text prompt is "{text_prompt}",
        all the frames of video are as follows:
        """
