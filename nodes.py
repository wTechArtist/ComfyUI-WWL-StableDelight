import os
import torch
import numpy as np
from PIL import Image
import folder_paths
from .stabledelight_comfyui.pipeline_yoso_delight import YosoDelightPipeline

class StableDelightNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "processing_resolution": ("INT", {
                    "default": 2048, 
                    "min": 512,
                    "max": 4096,
                    "step": 64
                }),
            }
        }

    CATEGORY = "image/delight"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "delight_image"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = YosoDelightPipeline.from_pretrained(
            'Stable-X/yoso-delight-v0-4-base', 
            trust_remote_code=True,
            variant="fp16",
            torch_dtype=torch.float16,
            t_start=0
        ).to(self.device)
        try:
            import xformers
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    def delight_image(self, image, processing_resolution):
        # Convert from ComfyUI image format to PIL
        image = (image * 255).clamp(0, 255)
        image = image.to(torch.uint8)
        if image.ndim == 4:
            image = image[0]
        image = Image.fromarray(image.cpu().numpy())

        # Process image
        pipe_out = self.pipe(
            image,
            match_input_resolution=False,
            processing_resolution=processing_resolution
        )
        
        # Convert output to ComfyUI format
        processed = pipe_out.prediction
        if isinstance(processed, np.ndarray):
            processed = torch.from_numpy(processed)
        if processed.ndim == 3:
            processed = processed.unsqueeze(0)
        processed = processed.float()
        
        return (processed,) 