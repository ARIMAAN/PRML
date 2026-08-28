from PIL import Image
import numpy as np

def rgb_to_grayscale(input_path, output_path):
    # Load image
    image = Image.open(input_path)

    # Make sure image is RGB
    image = image.convert("RGB")

    # Convert image to NumPy array
    rgb = np.array(image)

    # Extract RGB channels
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    # RGB → Grayscale formula
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert to 8-bit image
    gray = gray.astype(np.uint8)

    # Convert NumPy array back to PIL image
    gray_image = Image.fromarray(gray)

    # Save grayscale image
    gray_image.save(output_path)

    return gray