import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image


def predict(input_image, sess):
    # Preprocess the input image (e.g., resize, normalize)
    input_image = input_image.resize((256, 256))  # Adjust size as needed
    input_image = np.array(input_image).transpose(2, 0, 1)  # HWC to CHW
    input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0,1]
    input_image = (input_image - 0.5) / 0.5  # Normalize to [-1,1]
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Run the model
    inputs = {sess.get_inputs()[0].name: input_image}
    output = sess.run(None, inputs)

    # Post-process the output image (if necessary)
    output_image = output[0].squeeze().transpose(1, 2, 0)  # CHW to HWC
    output_image = (output_image + 1) / 2  # Scale to [0,1]
    output_image = (output_image * 255).astype(np.uint8)  # Denormalize to [0,255]

    return Image.fromarray(output_image)


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference on an image using an ONNX model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model file (e.g., sar2rgb.onnx)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image file (e.g., input.jpg)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output image (e.g., output.jpg)",
    )

    args = parser.parse_args()

    # Load the ONNX model
    sess = ort.InferenceSession(args.model)

    # Load the input image and ensure it's in RGB mode
    input_image = Image.open(args.input).convert("RGB")

    # Perform prediction
    output_image = predict(input_image, sess)

    # Save the output image
    output_image.save(args.output)
    print(f"Output image saved to {args.output}")


if __name__ == "__main__":
    main()
