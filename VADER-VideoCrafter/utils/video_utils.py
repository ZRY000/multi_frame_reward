import torch


try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True
# from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import math

import decord
import numpy as np
from decord import VideoReader, cpu

# from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision import transforms
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomHorizontalFlipVideo,
)
from transformers import ProcessorMixin
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

@torch.jit.ignore
def _interpolate_opencv(x: torch.Tensor, size: tuple[int, int], interpolation: str) -> torch.Tensor:
    """
    Down/up samples the input torch tensor x to the given size with given interpolation
    mode.
    Args:
        input (Tensor): the input tensor to be down/up sampled.
        size (Tuple[int, int]): expected output spatial size.
        interpolation: model to perform interpolation, options include `nearest`,
            `linear`, `bilinear`, `bicubic`.
    """
    if not _HAS_CV2:
        raise ImportError(
            'opencv is required to use opencv transforms. Please '
            "install with 'pip install opencv-python'.",
        )

    _opencv_pytorch_interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }
    assert interpolation in _opencv_pytorch_interpolation_map
    new_h, new_w = size
    img_array_list = [
        img_tensor.squeeze(0).numpy() for img_tensor in x.permute(1, 2, 3, 0).split(1, dim=0)
    ]
    resized_img_array_list = [
        cv2.resize(
            img_array,
            (new_w, new_h),  # The input order for OpenCV is w, h.
            interpolation=_opencv_pytorch_interpolation_map[interpolation],
        )
        for img_array in img_array_list
    ]
    img_array = np.concatenate(
        [np.expand_dims(img_array, axis=0) for img_array in resized_img_array_list],
        axis=0,
    )
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_array))
    img_tensor = img_tensor.permute(3, 0, 1, 2)
    return img_tensor


def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = 'bilinear',
    backend: str = 'pytorch',
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        An x-like Tensor with scaled spatial dims.
    """
    assert len(x.shape) == 4
    # assert x.dtype == torch.float32
    assert backend in ('pytorch', 'opencv')
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    if backend == 'pytorch':
        return torch.nn.functional.interpolate(
            x,
            size=(new_h, new_w),
            mode=interpolation,
            align_corners=False,
        )
    elif backend == 'opencv':
        return _interpolate_opencv(x, size=(new_h, new_w), interpolation=interpolation)
    else:
        raise NotImplementedError(f'{backend} backend not supported.')


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``short_side_scale``.
    """

    def __init__(self, size: int, interpolation: str = 'bilinear', backend: str = 'pytorch'):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return short_side_scale(
            x,
            self._size,
            self._interpolation,
            self._backend,
        )


def video_process(videos, target_frames=8):
    # videos = z_decode, shape: b, c, f, h, w
    frame_id_list = np.linspace(0, videos.shape[2] - 1, target_frames, dtype=int)
    video_outputs = videos[:,:,frame_id_list,:,:]
    video_outputs = ((video_outputs / 2) + 0.5).clamp(0, 1)
    transform = Compose(
        [
            # UniformTemporalSubsample(num_frames),
            # Lambda(lambda x: x / 255.0), # TODO: check if this is necessary
            NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ShortSideScale(size=224),
            CenterCropVideo(224),
            RandomHorizontalFlipVideo(p=0.5),
        ],
    )
    transformed_outputs = [transform(video_outputs[i]) for i in range(video_outputs.shape[0])]
    video_outputs = torch.stack(transformed_outputs)
    return video_outputs


def load_video_from_path(
    path: str,
    num_frames: int=8,
) -> torch.Tensor:
    decord.bridge.set_bridge('torch')
    decord_vr = VideoReader(path, ctx=cpu(0))
    duration = len(decord_vr)
    frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data

def image_process(videos):
    # videos = z_decode[:,:,id,:,:], shape: b, c, h, w
    video_outputs = ((videos / 2) + 0.5).clamp(0, 1)
    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),  # assume image
        ],
    )
    transformed_outputs = [transform(video_outputs[i]) for i in range(video_outputs.shape[0])]
    video_outputs = torch.stack(transformed_outputs)
    return video_outputs