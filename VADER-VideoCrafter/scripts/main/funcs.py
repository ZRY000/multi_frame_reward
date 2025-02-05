# Adapted from VideoCrafter: https://github.com/AILab-CVC/VideoCrafter
import os, sys, glob
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2
import random
from dataclasses import dataclass, field

import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_rg import RGDDIMSampler
from lvdm.models.samplers.ddim_vg import VGDDIMSampler
from lvdm.models.reward import (
    vllava_rew_fn,
)
import transformers
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
from transformers import Trainer
# import ipdb
# st = ipdb.set_trace

def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, backprop_mode=None, decode_frame='-1', **kwargs):
    ddim_sampler = DDIMSampler(model)
    if backprop_mode is not None:   # it is for training now, backprop_mode != None also means vader training mode
        ddim_sampler.backprop_mode = backprop_mode
        ddim_sampler.training_mode = True
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]

            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []

    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,              # samples: batch, c, t, h, w
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            **kwargs
                                            )
            
        ## reconstruct from latent to pixel space
        if backprop_mode is not None:   # it is for training now. Use one frame randomly to save memory
            try:
                decode_frame=int(decode_frame)
                #it's a int
            except:
                pass
            if type(decode_frame) == int:
                frame_index = random.randint(0,samples.shape[2]-1) if decode_frame == -1 else decode_frame        # samples: batch, c, t, h, w
                batch_images = model.decode_first_stage_2DAE(samples[:,:,frame_index:frame_index+1,:,:])
            elif decode_frame in ['alt', 'all']:
                idxs = range(0, samples.shape[2], 2) if decode_frame == 'alt' else range(samples.shape[2])
                batch_images = model.decode_first_stage_2DAE(samples[:,:,idxs,:,:])


        else:   # inference mode
            batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)

    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants

def rg_batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, backprop_mode=None, decode_frame='-1',\
                        decode_idxs=None, vae_scale_factor=None, neg_cond=None, **kwargs):
    ddim_sampler = RGDDIMSampler(model)
    if backprop_mode is not None:   # it is for training now, backprop_mode != None also means vader training mode
        ddim_sampler.backprop_mode = backprop_mode
        ddim_sampler.training_mode = True
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]

            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []

    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,              # samples: batch, c, t, h, w
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            neg_cond=neg_cond,
                                            **kwargs
                                            )
            
        ## reconstruct from latent to pixel space
        if backprop_mode is not None:   # it is for training now. Use one frame randomly to save memory
            try:
                decode_frame=int(decode_frame)
                #it's a int
            except:
                pass
            # if type(decode_frame) == int:
            #     frame_index = random.randint(0,samples.shape[2]-1) if decode_frame == -1 else decode_frame        # samples: batch, c, t, h, w
            #     batch_images = model.decode_first_stage_2DAE(samples[:,:,frame_index:frame_index+1,:,:])
            # elif decode_frame in ['alt', 'all']:
            #     idxs = range(0, samples.shape[2], 2) if decode_frame == 'alt' else range(samples.shape[2])
            #     batch_images = model.decode_first_stage_2DAE(samples[:,:,idxs,:,:])
            batch_images = decode(model, samples, batch_size, decode_idxs, vae_scale_factor)


        else:   # inference mode
            batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)

    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants

def vg_batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0, \
                        cfg_scale=1.0, temporal_cfg_scale=None, backprop_mode=None, z=None, \
                        start_idx=None, end_idx=None, is_last=None, decode_idxs=None, rg_type=None, vae_scale_factor=None, **kwargs):
    ddim_sampler = VGDDIMSampler(model)
    if backprop_mode is not None:   # it is for training now, backprop_mode != None also means vader training mode
        ddim_sampler.backprop_mode = backprop_mode
        ddim_sampler.training_mode = True
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]

            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    assert n_samples == 1
    
    if ddim_sampler is not None:
        kwargs.update({"clean_cond": True})
        z, intermediates = ddim_sampler.rollout(S=ddim_steps,              # samples: batch, c, t, h, w
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            z=z,
                                            start_idx=start_idx,
                                            end_idx=end_idx,
                                            **kwargs
                                            )
        # traj = torch.stack(intermediates['x_inter'][:-1])
        traj = intermediates['x_inter'][:-1]
    ## reconstruct from latent to pixel space
    # if backprop_mode is not None:   # it is for training now. Use one frame randomly to save memory
    #     try:
    #         decode_frame=int(decode_frame)
    #         #it's a int
    #     except:
    #         pass
    #     if type(decode_frame) == int:
    #         frame_index = random.randint(0,z.shape[2]-1) if decode_frame == -1 else decode_frame        # samples: batch, c, t, h, w
    #         z_decode = model.decode_first_stage_2DAE(z[:,:,frame_index:frame_index+1,:,:])
        
    #     elif decode_frame in ['alt', 'all']:
    #         idxs = range(0, z.shape[2], 2) if decode_frame == 'alt' else range(z.shape[2])
    #         z_decode = model.decode_first_stage_2DAE(z[:,:,idxs,:,:])

    # else:   # inference mode
    #     z_decode = model.decode_first_stage_2DAE(z)

    # z_decode = model.decode_first_stage_2DAE(z[:,:,decode_idxs,:,:])
    # with torch.no_grad():
    #     traj_decode = torch.stack([                                 # traj: l,b,c,f,h,w
    #         model.decode_first_stage_2DAE(traj[i][:,:,decode_idxs,:,:])
    #         for i in range(traj.shape[0])
    #     ])
        
    if is_last and rg_type=="refl":
        traj.append(z.clone().detach())
        forward_context = torch.autograd.graph.save_on_cpu
        with forward_context():
            step = ddim_sampler.ddim_timesteps[end_idx] 
            index = ddim_steps - end_idx - 1
            ts = torch.full((noise_shape[0],), step, device=ddim_sampler.model.betas.device , dtype=torch.long)
            outs = ddim_sampler.p_sample_ddim(z, cond, ts, index=index, use_original_steps=False,
                                        quantize_denoised=False, temperature=1.,
                                        noise_dropout=0., score_corrector=None,
                                        corrector_kwargs=None,
                                        unconditional_guidance_scale=cfg_scale,
                                        unconditional_conditioning=uc,
                                        x0=None,
                                        **kwargs)
        img, pred_x0 = outs
        z = pred_x0

    traj = torch.stack(traj)
    # z_decode = model.decode_first_stage_2DAE(z[:,:,decode_idxs,:,:])
    # with torch.no_grad():
    #     traj_decode = torch.stack([                                 # traj: l,b,c,f,h,w
    #         model.decode_first_stage_2DAE(traj[i][:,:,decode_idxs,:,:])
    #         for i in range(traj.shape[0])
    #     ])
    z_decode = decode(model, z, batch_size, decode_idxs, vae_scale_factor)  # z_decode: b,c,f,h,w
    with torch.no_grad():
        traj_decode = torch.stack([                                 # traj: l,b,c,f,h,w
            decode(model, traj[i], batch_size, decode_idxs, vae_scale_factor)
            for i in range(traj.shape[0])
        ])
    # print(f"z_decode shape: {z_decode.shape}")
    # print(f"traj_decode shape: {traj_decode.shape}")
    return z, z_decode, traj_decode

def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

from PIL import Image
def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def get_critic_optimizer(args, model):
    critic_name = args.critic_name
    critic_type = args.critic_type


    if critic_name == "hps_clip" and critic_type == "single-frame":
        ciritc_args = {
            "lr": 0.0000033,
            "weight_decay": 0.35
        }

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        critic_optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": ciritc_args["weight_decay"]}, # TODO: check if it is correct
            ],
            lr=ciritc_args["lr"],
        )

    elif critic_name == "languagebind_video" and critic_type == "multi-frame": # copied from LanguageBind
        ciritc_args = {
            "lr": 5e-4,
            "weight_decay": 0.2,
            "coef_lr": 1e-3
        }

        no_decay = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n or 'class_embedding' in n or 'patch_embedding' in n
        decay = lambda n, p: not no_decay(n, p)

        lora = lambda n, p: "lora" in n
        non_lora = lambda n, p: not lora(n, p)

        named_parameters = list(model.named_parameters())
        no_decay_non_lora_params = [[n, p] for n, p in named_parameters if no_decay(n, p) and non_lora(n, p) and p.requires_grad]
        decay_non_lora_params = [[n, p] for n, p in named_parameters if decay(n, p) and non_lora(n, p) and p.requires_grad]

        no_decay_lora_params = [[n, p] for n, p in named_parameters if no_decay(n, p) and lora(n, p) and p.requires_grad]
        decay_lora_params = [[n, p] for n, p in named_parameters if decay(n, p) and lora(n, p) and p.requires_grad]

        param_groups = []
        if no_decay_non_lora_params:
            param_groups.append({
                "params": [p for n, p in no_decay_non_lora_params],
                "weight_decay": 0.,
                'lr': ciritc_args["lr"] * ciritc_args["coef_lr"]
            })
        if decay_non_lora_params:
            param_groups.append({
                "params": [p for n, p in decay_non_lora_params],
                "weight_decay": ciritc_args["weight_decay"],
                'lr': ciritc_args["lr"] * ciritc_args["coef_lr"]
            })
        if no_decay_lora_params:
            param_groups.append({
                "params": [p for n, p in no_decay_lora_params],
                "weight_decay": 0.
            })
        if decay_lora_params:
            param_groups.append({
                "params": [p for n, p in decay_lora_params],
                "weight_decay": ciritc_args["weight_decay"]
            })

        critic_optimizer = torch.optim.AdamW(
            param_groups,
            lr=ciritc_args["lr"],
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    elif critic_name == "video_llava" and critic_type == "multi-frame":
        critic_args = {
            "lr": 2e-5,
            "weight_decay": 0.1,
            "mm_projector_lr": None
        }
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if 'bias' not in name]
        if critic_args["mm_projector_lr"] is not None:
            projector_parameters = [
                name for name, _ in model.named_parameters() if 'mm_projector' in name
            ]
            optimizer_grouped_parameters = [
                {
                    'params': [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n in decay_parameters
                            and n not in projector_parameters
                            and p.requires_grad
                        )
                    ],
                    'weight_decay': critic_args["weight_decay"],
                },
                {
                    'params': [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n not in decay_parameters
                            and n not in projector_parameters
                            and p.requires_grad
                        )
                    ],
                    'weight_decay': 0.0,
                },
                {
                    'params': [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n in decay_parameters
                            and n in projector_parameters
                            and p.requires_grad
                        )
                    ],
                    'weight_decay': critic_args["weight_decay"],
                    'lr': critic_args["mm_projector_lr"],
                },
                {
                    'params': [
                        p
                        for n, p in model.named_parameters()
                        if (
                            n not in decay_parameters
                            and n in projector_parameters
                            and p.requires_grad
                        )
                    ],
                    'weight_decay': 0.0,
                    'lr': critic_args["mm_projector_lr"],
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    'params': [
                        p
                        for n, p in model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    'weight_decay': critic_args["weight_decay"],
                },
                {
                    'params': [
                        p
                        for n, p in model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    'weight_decay': 0.0,
                },
            ]
        # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(critic_args)

        # critic_optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        critic_optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=critic_args["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
        )

    else:
        raise NotImplementedError(f"Critic {critic_name}, {critic_type} is not implemented.")

    return critic_optimizer

import numpy as np

class CriticDataset:
    def __init__(self, max_size=8):
        self.traj_ds = []
        self.target_values_ds = []
        self.end_state_ds = []
        self.rew_buf_ds = []
        self.is_last_ds = []
        self.max_size = max_size

    def update_traj(self, traj, end_state, rew_buf, is_last):
        self.traj_ds.append(traj)
        self.end_state_ds.append(end_state)
        self.rew_buf_ds.append(rew_buf)
        self.is_last_ds.append(is_last)
        self.target_values_ds = []  # clear all history target values, new target values are computed by the new target critic

        if len(self.traj_ds) > self.max_size:
            self.traj_ds = self.traj_ds[1:]
            self.end_state_ds = self.end_state_ds[1:]
            self.rew_buf_ds = self.rew_buf_ds[1:]
            self.is_last_ds = self.is_last_ds[1:]


    @torch.no_grad()
    def update_target_values(self, target_critic, prompts, critic_method, gamma, lam):
        for i,traj in enumerate(self.traj_ds):
            target_values = compute_target_values(
                target_critic=target_critic,
                rew_buf=self.rew_buf_ds[i],
                traj_decode=traj,
                z_decode=self.end_state_ds[i],
                prompts=prompts,
                critic_method=critic_method,
                gamma=gamma,
                lam=lam,
                is_last=self.is_last_ds[i]
            )
            self.target_values_ds.append(target_values)
        

    def shuffle(self):
        # Generate a permutation of indices and apply it to both traj_ds and target_values_ds
        indices = np.random.permutation(len(self.traj_ds))
        self.traj_ds = [self.traj_ds[i] for i in indices]
        self.target_values_ds = [self.target_values_ds[i] for i in indices]
        self.end_state_ds = [self.end_state_ds[i] for i in indices]
        self.rew_buf_ds = [self.rew_buf_ds[i] for i in indices]
        self.is_last_ds = [self.is_last_ds[i] for i in indices]

    def ds_tensor(self):
        return torch.stack(self.traj_ds), torch.stack(self.target_values_ds)

    def __len__(self):
        return len(self.traj_ds)

    def __getitem__(self, index):
        return {'traj': self.traj_ds[index], 'target_values': self.target_values_ds[index]}



@torch.no_grad()
def compute_target_values(target_critic, rew_buf, traj_decode, z_decode, prompts, critic_method, gamma, lam, is_last=False):
    #   TODO: verify
    dtype , device  = z_decode.dtype , z_decode.device


    #   traj_len: the length of the trajectory
    traj_len      = traj_decode.shape[0]

    done_mask     = torch.zeros(traj_len, dtype = dtype, device = device)
    next_values   = torch.zeros(traj_len, z_decode.shape[0], dtype = dtype, device = device)
    target_values = torch.zeros(traj_len, z_decode.shape[0], dtype = dtype, device = device)

    '''
    z_decode shape : (bsz,n_frame,channels,height,width)
    rew_buf: the buffer of rewards in a window, reward is not zero if and only if it's the state before the last state
    rew_buf = torch.zeros(traj_len, z_decode.shape[0], dtype = dtype, device = device)
    gamma : 0.99
    lam : 0.95
    '''


    #   next_values: the value of the next state
    for i in range(traj_len - 1):
        next_values[i] = target_critic.forward(traj_decode[i+1], prompts)

    # TODO: DONE ： whether the value of the last state is 0
    next_values[traj_len - 1] = target_critic.forward(z_decode, prompts) if not is_last else torch.zeros(z_decode.shape[0], dtype=dtype,device=device)



    #  done_mask:
    done_mask[traj_len - 1] = 1

    #   target_values

    if critic_method == 'one-step':

        target_values = rew_buf + gamma * next_values
    elif critic_method == 'td-lambda' and lam != 1:

        Ai      = torch.zeros(z_decode.shape[0], dtype = dtype, device = device)
        Bi      = torch.zeros(z_decode.shape[0], dtype = dtype, device = device)
        lam_acc = torch.ones(z_decode.shape[0],  dtype = dtype, device = device)

        for i in reversed(range(traj_len)):     #   TODO: verify "traj_len" or "traj_len"
            lam_acc = lam_acc * lam * (1. - done_mask[i]) + done_mask[i]

            Ai = (1.0 - done_mask[i]) * (lam * gamma * Ai + gamma * next_values[i] + (1. - lam_acc) / (1. - lam) * rew_buf[i])

            Bi = gamma * (next_values[i] * done_mask[i] + Bi * (1.0 - done_mask[i])) + rew_buf[i]

            target_values[i] = (1.0 - lam) * Ai + lam_acc * Bi
            
    elif critic_method == 'td-lambda' and lam == 1:

        Bi      = torch.zeros(z_decode.shape[0], dtype = dtype, device = device)

        for i in reversed(range(traj_len)):     

            Bi = gamma * (next_values[i] * done_mask[i] + Bi * (1.0 - done_mask[i])) + rew_buf[i]

            target_values[i] = Bi

    else:
        raise NotImplementedError

    return target_values

# TODO: modify DONE
def compute_actor_loss(target_critic, reward, end_idx, traj_len, z_decode, prompts, gamma, loss_type, is_last=False):

    if loss_type == "whole":
        if is_last:
            actor_loss = (gamma ** (end_idx - 1)) * (reward.mean())
        else:
            actor_loss = (gamma ** end_idx) * (target_critic(z_decode, prompts).mean())

    elif loss_type == "horizon":
        if is_last:
            actor_loss = (gamma ** (traj_len - 1)) * (reward.mean())
        else:
            actor_loss = (gamma ** traj_len) * (target_critic(z_decode, prompts).mean())

    elif loss_type == "same":
        if is_last:
            actor_loss = reward.mean()
        else:
            actor_loss = target_critic(z_decode, prompts).mean()

    elif loss_type == "distill":
        assert is_last == False
        actor_loss = target_critic(z_decode, prompts).mean()

    else:
        raise NotImplementedError(f"Loss type {loss_type} is not implemented for value gradient.")

    return 1-actor_loss

def compute_critic_loss(critic, traj, target_values, prompts):
    # Predict values 
    predicted_values = torch.stack([
        critic(traj[i],prompts)
        for i in range(traj.shape[0])
    ])

    # Ensure that predicted_values and target_values have the same shape
    assert predicted_values.shape == target_values.shape, \
        f"Shape mismatch: {predicted_values.shape} vs {target_values.shape}"

    # Calculate mean squared error loss
    critic_loss = ((predicted_values - target_values) ** 2).mean()

    return critic_loss

def compute_KL(model, pre_model, cond, noise_shape, ddim_steps=50, ddim_eta=1.0,\
            cfg_scale=1.0, temporal_cfg_scale=None, \
            z=None, start_idx=None, rollout_length=None, **kwargs):
    ddim_sampler = VGDDIMSampler(model)
    
    ddim_sampler.training_mode = True
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]

            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    if ddim_sampler is not None:
        kwargs.update({"clean_cond": True})
        z, intermediates, end_idx = ddim_sampler.rollout(S=ddim_steps,              # samples: batch, c, t, h, w
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            z=z,
                                            start_idx=start_idx,
                                            rollout_length=rollout_length,
                                            **kwargs
                                            )
        traj = torch.stack(intermediates['x_inter'][:-1])
        
    return z, traj, z_decode, traj_decode, end_idx

# def get_rew_fn(fn_name, model_path, device, **kwargs,):
#     if fn_name == "vllava":
#         rew_fn = vllava_rew_fn(model_path, device, **kwargs)
#     return rew_fn

def decode(model, z, batch_size, decode_idxs, vae_scale_factor):
    decode_bs = 1
    b_idx = torch.randperm(batch_size)[: decode_bs]
    z_selected = (
        z[b_idx][:, :, decode_idxs] / vae_scale_factor
    )
    z_selected = z_selected.permute(0, 2, 1, 3, 4)
    z_selected = z_selected.reshape(
        len(decode_idxs), *z_selected.shape[2:]
    )
    z_decode = model.first_stage_model.decode(z_selected.to(model.first_stage_model.dtype))
    # z_decode = (z_decode / 2 + 0.5).clamp(0, 1)
    z_decode = z_decode.reshape(
        batch_size, len(decode_idxs), *z_decode.shape[1:]
    )
    z_decode = z_decode.permute(0, 2, 1, 3, 4)

    return z_decode

def print_optimizer_params(optimizer, print_info):
    total_params = 0
    trainable_params = 0

    # Iterate through each parameter group in the optimizer
    for param_group in optimizer.param_groups:
        # if 'name' in param_group and print_info:
            # print(f"  Name: {param_group['name']}")
        for param in param_group['params']:
            # if print_info:
            #     print(f"  - Parameter size: {param.size()}")
            # Count the total number of parameters
            total_params += param.numel()  # numel() gives the number of elements in the tensor
            
            # Check if the parameter requires gradient (trainable)
            if param.requires_grad:
                trainable_params += param.numel()

    # Print out the results
    if print_info:
        print(f"Total number of parameters: {total_params}")
        print(f"Number of trainable parameters: {trainable_params}")
