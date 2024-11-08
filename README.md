# StableDelight ComfyUI Node

## Installation

1. Install ComfyUI following the instructions at [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

2. Clone this repository into ComfyUI's custom_nodes directory:
```
git clone https://github.com/Stable-X/StableDelight.git
```

## Torch Hub Loader 🚀
To use the StableDelight pipeline, you can instantiate the model and apply it to an image as follows:

```python
import torch
from PIL import Image

# Load an image
input_image = Image.open("path/to/your/image.jpg")

# Create predictor instance
predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)

# Apply the model to the image
delight_image = predictor(input_image)

# Save or display the result
delight_image.save("output/delight.png")
```

## Gradio interface 🤗

We also provide a Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> interface for a better experience, just run by:

```bash
# For Linux and Windows users (and macOS with Intel??)
python app.py
```

You can specify the `--server_port`, `--share`, `--server_name` arguments to satisfy your needs!


