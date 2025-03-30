"""
Inference Script
"""

from pathlib import Path

import torch
from torchvision.transforms import v2

from PIL import Image

from utils.config import Config
from src.pix2pix import Pix2Pix


def main():
    print("Starting inference...")
    # Load configuration
    config = Config("config.yaml")
    # Set device
    device = torch.device(config["inference"]["device"])
    # Create model
    model = (
        Pix2Pix(
            c_in=config["model"]["c_in"],
            c_out=config["model"]["c_out"],
            is_train=False,
            use_upsampling=config["model"]["use_upsampling"],
            mode=config["model"]["mode"],
        )
        .to(device)
        .eval()
    )

    gen_checkpoint = Path(config["inference"]["gen_checkpoint"])
    if not gen_checkpoint.exists():
        raise FileNotFoundError(
            f"Generator checkpoint file not found: {gen_checkpoint}\nPlease check config.yaml"
        )

    model.load_model(gen_path=gen_checkpoint)

    img_path = Path(config["inference"]["image_path"])

    if not img_path.exists() and not img_path.is_file():
        raise FileNotFoundError(
            f"A valid image file not found: {img_path}\nPlease check config.yaml"
        )

    img = Image.open(img_path).convert("RGB")

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    img = transforms(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model.generate(img, is_scaled=True)
        pred = torch.clamp(pred, -1, 1)
        pred = (pred + 1) / 2.0
        pred = (pred * 255).to(torch.uint8)
        pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    output_path = Path(config["inference"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = Image.fromarray(pred)
    output.save(output_path)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
