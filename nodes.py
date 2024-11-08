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

    def delight_image(self, image, processing_resolution):
        # Convert from ComfyUI image format to PIL and resize
        if torch.is_tensor(image):
            if image.ndim == 4:
                image = image[0]
            image = (image * 255).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(image.cpu().numpy())
        
        # Resize image
        image = self.resize_image(image, processing_resolution)
        
        # Process image
        pipe_out = self.pipe(
            image,
            match_input_resolution=False,
            processing_resolution=processing_resolution
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