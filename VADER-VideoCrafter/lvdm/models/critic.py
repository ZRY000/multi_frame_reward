import torch, torchvision
from HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import torch.nn as nn
from lvdm.models.video import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoConfig
from utils.video_utils import video_process
import json
from transformers import AutoConfig
import torch.utils.checkpoint as checkpoint


# Critic is a clip model, initialized by the hps_v2.1_compressed weights
# we may need to consider video models for critic
class Critic(nn.Module):
    def __init__(self, args, dtype, device, ckp_path=None):
        super().__init__()
        # critic = Critic(cfg, dtype, accelerator.device)
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

        hps_version = args.hps_version
        if ckp_path is not None:
            checkpoint_path = ckp_path
        elif hps_version == "v2.0":
            checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2_compressed.pt"
        else:   # hps_version == "v2.1"
            checkpoint_path = f"/home/juntao/Models/HPSv2/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if ckp_path is not None:
            # self.model.load_state_dict(checkpoint)
            self.load_state_dict(checkpoint)
            print(f"Loading from {checkpoint_path}")
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        print("hpsv2 successfully loaded")

        self.model = self.model.to(device, dtype=dtype)

        self.model.logit_scale.requires_grad = False
        self.tokenizer.requires_grad_(False)

    def forward(self, im_pix, prompts): # might need modification
        # the computation should be the same as reward functions
        target_size =  (224, 224)
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(self.dtype)
        caption = self.tokenizer(prompts)
        caption = caption.to(self.device)
        outputs = self.model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        outputs["logit_scale"].detach()    # add to debug
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)

        return scores

class MultiFrameCritic(nn.Module): # TODO: sync with single-frame critic
    def __init__(self, dtype, device, ckp_path=None):
        super().__init__()
        self.pretrained_ckpt = '/home/juntao/Models/LanguageBind/LanguageBind_Video_FT'
        # self.config = '/home/juntao/Projects/zry/VADER/VADER-VideoCrafter/configs/mf_critic_config.json'
        # self.model = LanguageBindVideo.from_pretrained(self.pretrained_ckpt, config=self.config, cache_dir='./cache_dir')
        self.model = LanguageBindVideo.from_pretrained(self.pretrained_ckpt, cache_dir='./cache_dir')
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(self.pretrained_ckpt, cache_dir='./cache_dir')
        self.embed_size = self.model.projection_dim
        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=8, batch_first=True)
        # Linear Layer for Score Prediction
        self.fc = nn.Linear(self.embed_size, 1)  # Output a single scalar

        if ckp_path is not None:
            checkpoint = torch.load(ckp_path, map_location=device)
            self.load_state_dict(checkpoint)
            print(f"Loading from {ckp_path}")

        self.dtype = dtype
        self.device = device

        self.model = self.model.to(device, dtype=dtype)
        self.model.logit_scale.requires_grad_(False)

    # TODO: verify!
    def forward(self, videos, prompts):
        text_encoding = self.tokenizer(
            prompts,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        # videos = videos.to(torch.float32)
        video_outputs = video_process(videos, target_frames=8).to(self.dtype)
        out = checkpoint.checkpoint(
            self.model,
            input_ids=text_encoding.input_ids.to(self.device),
            pixel_values=video_outputs,
            attention_mask=text_encoding.attention_mask.to(self.device),
            use_reentrant=False
        )
        # out = self.model(
        #     input_ids=text_encoding.input_ids.to(self.device),
        #     pixel_values=video_outputs,
        #     attention_mask=text_encoding.attention_mask.to(self.device),
        # )
        
        # logits = out.text_embeds @ out.image_embeds.T
        # scores = torch.diagonal(logits)
        
        # Add sequence dimension to embeddings (required for attention)
        image_embeds = out.image_embeds.unsqueeze(1)  # Shape: (batch_size, 1, embed_size)
        text_embeds = out.text_embeds.unsqueeze(1)    # Shape: (batch_size, 1, embed_size)
        
        # Cross Attention: Query = image, Key/Value = text
        # Output: attended_image_embeds has shape (batch_size, 1, embed_size)
        attended_image_embeds, _ = self.cross_attention(query=image_embeds, key=text_embeds, value=text_embeds)
        
        # Remove sequence dimension
        attended_image_embeds = attended_image_embeds.squeeze(1)  # Shape: (batch_size, embed_size)
        
        # Linear layer to predict score
        scores = self.fc(attended_image_embeds)  # Shape: (batch_size, 1)
        return scores.squeeze(-1)
