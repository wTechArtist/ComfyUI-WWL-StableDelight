import os
import torch
import numpy as np
from PIL import Image
import folder_paths
from .stabledelight_comfyui.pipeline_yoso_delight import YosoDelightPipeline

class StableDelightNode:
    upscale_methods = ["bilinear", "nearest-exact", "area", "bicubic"]
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
                "t_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "match_input_resolution": ("BOOLEAN", {
                    "default": False
                }),
                "ensemble_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff
                }),
                "resample_method": (s.upscale_methods,),
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

    def resize_image(self, input_image, resolution):
        # Convert tensor to PIL Image
        if torch.is_tensor(input_image):
            if input_image.ndim == 4:
                input_image = input_image[0]
            input_image = (input_image * 255).clamp(0, 255).to(torch.uint8)
            input_image = Image.fromarray(input_image.cpu().numpy())
        
        # Get image dimensions
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        
        # Resize the image
        return input_image.resize((W, H), Image.Resampling.LANCZOS)

    def delight_image(self, image, processing_resolution, t_start, match_input_resolution, 
                     ensemble_size, batch_size, seed=-1, resample_method="lanczos"):
        # Set random seed if provided
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            
        # Convert from ComfyUI image format to PIL and resize
        if torch.is_tensor(image):
            if image.ndim == 4:
                image = image[0]
            image = (image * 255).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(image.cpu().numpy())
        
        # Process image
        pipe_out = self.pipe(
            image,
            match_input_resolution=match_input_resolution,
            processing_resolution=processing_resolution,
            t_start=t_start,
            ensemble_size=ensemble_size,
            batch_size=batch_size,
            generator=generator,
            resample_method_input=resample_method,
            resample_method_output=resample_method
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