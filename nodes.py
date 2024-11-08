import os
import torch
import numpy as np
from PIL import Image
import folder_paths
from .stabledelight_comfyui.pipeline_yoso_delight import YosoDelightPipeline

class StableDelightNode:
    resample_methods = ["bilinear","nearest", "nearest-exact",  "bicubic", "area"]
    
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
                "resample_method": (s.resample_methods,),
                "controlnet_conditioning_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
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

    def delight_image(self, image, processing_resolution, resample_method, controlnet_conditioning_scale):
        # Convert from ComfyUI image format to PIL
        if torch.is_tensor(image):
            if image.ndim == 4:
                image = image[0]
            image = (image * 255).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(image.cpu().numpy())
        
        # Process image
        pipe_out = self.pipe(
            image,
            processing_resolution=processing_resolution,
            resample_method_input=resample_method,
            resample_method_output=resample_method,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        )
        
        # Convert output to ComfyUI format
        processed = pipe_out.prediction
        processed = (processed.clip(-1, 1) + 1) / 2
        processed = (processed[0] * 255).astype(np.uint8)
        
        # Convert to tensor format expected by ComfyUI
        processed = torch.from_numpy(processed).float() / 255.0
        if processed.ndim == 3:
            processed = processed.unsqueeze(0)
        
        return (processed,)