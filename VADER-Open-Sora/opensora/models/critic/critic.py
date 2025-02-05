import torch, torchvision
from HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

# Critic is a clip model, initialized by the hps_v2.1_compressed weights
# we may need to consider video models for critic
class Critic():
    def __init__(self, cfg, dtype, peft_model_dtype, device):
        # critic = Critic(cfg, dtype, peft_model.dtype, accelerator.device)
        self.dtype = dtype
        self.device = device
        # we need to build the modules of the hps model, so that we can use the parameters to optimize
        self.model_name = "ViT-H-14"

        self.model, preprocess_train, preprocess_val = create_model_and_transforms(
                self.model_name,
                "/home/juntao/Models/HPSv2/HPS_v2.1_compressed.pt", # this should be the path to the checkpoint?
                precision=dtype,
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

        self.tokenizer = get_tokenizer(self.model_name)

        hps_version = cfg.get("hps_version", "v2.1")
        if hps_version == "v2.0":
            checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2_compressed.pt"
        else:   # hps_version == "v2.1"
            checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("hpsv2 successfully loaded \n hpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \nhpsv2 successfully loaded \n")

        self.model = self.model.to(device, dtype=dtype)
        self.lam = cfg.get("lam", 0.5)



    def forward(self, **kwargs): 
        # **kwargs: the selected frame of the vae-decoded intermediate denoised samples
        # **kwargs: state_frames, state_prompt
        # allow multiple images to be passed into computation, but only receive one prompt as condition
        # the computation should be the same as reward functions
        state_frames = kwargs['state_frames']
        state_prompt = kwargs['state_prompt']
        target_size =  (224, 224)
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        state_frames = ((state_frames / 2) + 0.5).clamp(0, 1)
        x_var = torchvision.transforms.Resize(target_size)(state_frames)
        x_var = normalize(x_var).to(self.dtype)
        caption = self.tokenizer(state_prompt)
        caption = caption.to(self.device)
        outputs = self.model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        # logits = image_features @ text_features.T
        # scores = torch.diagonal(logits)
        logits = [image_feature.T @ text_features for image_feature in image_features]
        # we may need to adjust the computation for batch prompts

        return logits

    
    def loss(self, critic_dataset, prompt, estimated_values): #TODO: add it
        # self: critic
        # we need 
        # 1.critic dataset: N trajectories in the window of length l, size: N, l, c, h, w
        # 2.prompt
        # 3.estimated_values: the estimated values are computed using target_critic and the reward
        predicted_values = self.forward(critic_dataset.squeeze(), prompt) # we may need to transform the size of the citic_dataset
        critic_loss = ((predicted_values - estimated_values) ** 2).mean()
        return critic_loss

