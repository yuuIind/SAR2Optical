from pathlib import Path

import torch

from utils.config import Config
from src.pix2pix import Pix2Pix


def main():
    print("Starting Conversion...")
    config = Config("config.yaml")
    # Set device
    device = torch.device("cpu")
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

    opset_version = config["export"]["onnx"]["opset_version"]
    input_name = "input"
    output_name = "output"

    output_path = config["export"]["export_path"]
    input_shape = config["export"]["input_shape"]

    dummy_input = torch.randn(input_shape, requires_grad=True)
    _ = model(dummy_input)

    if config["export"]["is_dynamic"]:
        print("Exporting dynamic ONNX model...")
        dynamic_axes = {input_name: {0: "N"}}
        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            output_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=opset_version,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=[input_name],  # the model's input names
            output_names=[output_name],  # the model's output names
            dynamic_axes=dynamic_axes,  # variable length axes
        )

    else:
        print("Exporting static ONNX model...")
        torch.onnx.export(
            model,  # model being run
            dummy_input,  # model input (or a tuple for multiple inputs)
            output_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=opset_version,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=[input_name],  # the model's input names
            output_names=[output_name],  # the model's output names
        )


if __name__ == "__main__":
    main()
