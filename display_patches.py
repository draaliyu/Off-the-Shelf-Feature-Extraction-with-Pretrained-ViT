# display_patches.py
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from vit_keras import vit
from tkinter import Tk, filedialog

IMAGE_SIZE = 224
PATCH_SIZE = 16  # ViT typically uses 16x16 patches


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
    return image


def extract_patches(image, patch_size):
    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, 0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, (-1, patch_size, patch_size, 3))
    return patches


def display_patches(patches):
    num_patches = patches.shape[0]
    grid_size = int(np.sqrt(num_patches))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = axes.flatten()

    for img, ax in zip(patches, axes):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Create a Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )

    if image_path:
        image = load_image(image_path)
        patches = extract_patches(image, PATCH_SIZE)
        display_patches(patches)
    else:
        print("No image selected.")


if __name__ == '__main__':
    main()
