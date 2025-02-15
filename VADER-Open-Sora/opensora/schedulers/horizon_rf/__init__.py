# adapted from Open-Sora: https://github.com/hpcaitech/Open-Sora
import torch
from tqdm import tqdm
import random
import torch.utils.checkpoint as checkpoint

from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform




@SCHEDULERS.register_module("horizon_rflow") #TODO:change the name, already changed
class HRFLOW: #TODO:change the class name, already changed
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000, # same as the training num_timesteps 
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )
    
    def sample(
        self,
        model,
        text_encoder,
        z, 
        prompts,
        device,
        horizon,
        w_idx, 
        window,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        backprop_mode=None, 
        use_grad_checkpoint=False,
    ):  

        # TODO: set the horizon length
        # horizon = 5

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps] # multiply the timesteps by batchsize
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]
        
        
        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)

        if backprop_mode == 'last':
            backprop_cutoff_idx = self.num_sampling_steps - 1
        elif backprop_mode == 'rand':
            backprop_cutoff_idx = random.randint(0, self.num_sampling_steps - 1)
        elif backprop_mode == 'specific':
            backprop_cutoff_idx = 15
        elif backprop_mode == None:
            backprop_cutoff_idx = self.num_sampling_steps + 1   # no vader backprop
        elif backprop_mode == 'full':
            backprop_cutoff_idx = 0 # TODO: add 'full' to the backprop_mode
        else:
            raise ValueError(f"Unknown backprop_mode: {backprop_mode}")

        # TODO: rollout
        # TODO: sample N times -- to be added(add as the first dimension of z)
        def rollout(z, w_idx, window): 
            # z is the input latent
            # w_idx is the index of the window
            # window is the list of timesteps of the w_idx item in the list windows(window = windows[w_idx])
            if mask is not None:
                noise_added = torch.zeros_like(mask, dtype=torch.bool)
                noise_added = noise_added | (mask == 1)
            traj = []
            for i, t in progress_wrap(enumerate(window)):
                traj.append(z)
                if backprop_mode != None:   # if backprop_mode is None, no vader backprop, so it will not interfere with original opensora code
                    if i >= backprop_cutoff_idx:
                        for name, param in model.named_parameters():
                            if "lora" in name:
                                param.requires_grad = True
                    else:
                        model.requires_grad_(False)

                # mask for adding noise
                if mask is not None:
                    mask_t = mask * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                    mask_t_upper = mask_t >= t.unsqueeze(1)
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added

                    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                    noise_added = mask_t_upper

                # classifier-free guidance
                z_in = torch.cat([z, z], 0)
                t = torch.cat([t, t], 0)
                # apply gradient checkpointing to save memory during backpropagation
                if use_grad_checkpoint: #TODO: might need modification
                    pred = checkpoint.checkpoint(model, z_in, t, **model_args, use_reentrant=False).chunk(2, dim=1)[0]
                else:
                    pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
                    
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # update z
                # map the window_index to the timesteps_index
                j = i + horizon * w_idx
                dt = timesteps[j] - timesteps[j + 1] if j < len(timesteps) - 1 else timesteps[j]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt[:, None, None, None, None]

                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
            
            return z, traj

        # TODO: for short horizon
        # TODO: rollout() one horizon input/output
        
        z, traj = rollout(z, w_idx, window)

        return z, traj

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

    def get_windows(self, z, horizon, device, additional_args=None):
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps] # multiply the timesteps by batchsize
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]
        
        # TODO: slice the timesteps into sub-windows
        windows = []
        for i in range(self.num_sampling_steps // horizon + 1):
            if horizon * i <= self.num_sampling_steps - 1:
                windows.append(timesteps[horizon * i : horizon * (i+1)]) 

        return windows