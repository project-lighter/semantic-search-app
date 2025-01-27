from PIL import Image, ImageEnhance, ImageDraw
import streamlit as st
import numpy as np
import cv2
import torch
from torch.nn import Module
from typing import Dict, List
import matplotlib.pyplot as plt
from loguru import logger


def make_fig(image, preds, point_axs=None, current_idx=None, view=None, patch_size=None):
    # Convert A to an image
    image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Create a yellow mask from B
    if preds is not None:
        mask = np.where(preds == 1, 255, 0).astype(np.uint8)
        mask = Image.merge("RGB", 
                           (Image.fromarray(mask), 
                            Image.fromarray(mask), 
                            Image.fromarray(np.zeros_like(mask, dtype=np.uint8))))

        # Overlay the mask on the image
        image = Image.blend(image.convert("RGB"), mask, alpha=st.session_state.transparency)
    
    if point_axs is not None:
        draw = ImageDraw.Draw(image)

        if patch_size is not None:
            radius = patch_size[-1]//2
        else:
            radius = 32

        z, y, x = point_axs
        if z == current_idx:
            draw.rectangle((x-radius, y-radius, x+radius, y+radius), fill=None, width=3, outline="#2909F1")
    return image


def adjust_prefix_and_load_state_dict(
    model: Module, ckpt_path: str, ckpt_to_model_prefix: Dict[str, str] = None, layers_to_ignore: List[str] = None
) -> Module:
    """Load state_dict from a checkpoint into a model using `torch.load(strict=False`).
    `ckpt_to_model_prefix` mapping allows to rename the prefix of the checkpoint's state_dict keys
    so that they match those of the model's state_dict. This is often needed when a model was trained
    as a backbone of another model, so its state_dict keys won't be the same to those of a standalone
    version of that model. Prior to defining the `ckpt_to_model_prefix`, it is advised to manually check
    for mismatch between the names and specify them accordingly.

    Args:
        model (Module): PyTorch model instance to load the state_dict into.
        ckpt_path (str): Path to the checkpoint.
        ckpt_to_model_prefix (Dict[str, str], optional): A dictionary that maps keys in the checkpoint's
            state_dict to keys in the model's state_dict. If None, no key mapping is performed. Defaults to None.
        layers_to_ignore (List[str], optional): A list of layer names that won't be loaded into the model.
            Specify the names as they are after `ckpt_to_model_prefix` is applied. Defaults to None.
    Returns:
        The model instance with the state_dict loaded.

    Raises:
        ValueError: If there is no overlap between checkpoint's and model's state_dict.
    """

    # Load checkpoint
    ckpt = torch.load(ckpt_path)

    # Check if the checkpoint is a model's state_dict or a LighterSystem checkpoint.
    # A LighterSystem checkpoint contains the model’s entire internal state, we only need its state_dict.
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
        # Remove the "model." prefix from the checkpoint's state_dict keys. This is characteristic to LighterSystem.
        ckpt = {key.replace("model.", ""): value for key, value in ckpt.items()}

    # Adjust the keys in the checkpoint's state_dict to match the the model's state_dict's keys.
    if ckpt_to_model_prefix is not None:
        for ckpt_prefix, model_prefix in ckpt_to_model_prefix.items():
            # Add a dot at the end of the prefix if necessary.
            ckpt_prefix = ckpt_prefix if ckpt_prefix == "" or ckpt_prefix.endswith(".") else f"{ckpt_prefix}."
            model_prefix = model_prefix if model_prefix == "" or model_prefix.endswith(".") else f"{model_prefix}."
            if ckpt_prefix != "":
                # Replace ckpt_prefix with model_prefix in the checkpoint state_dict
                ckpt = {key.replace(ckpt_prefix, model_prefix): value for key, value in ckpt.items() if ckpt_prefix in key}
            else:
                # Add the model_prefix before the current key name if there's no specific ckpt_prefix
                ckpt = {f"{model_prefix}{key}": value for key, value in ckpt.items() if ckpt_prefix in key}

    # Check if there is no overlap between the checkpoint's and model's state_dict.
    if not set(ckpt.keys()) & set(model.state_dict().keys()):
        raise ValueError(
            "There is no overlap between checkpoint's and model's state_dict. Check their "
            "`state_dict` keys and adjust accordingly using `ckpt_prefix` and `model_prefix`."
        )

    # Remove the layers that are not to be loaded.
    if layers_to_ignore is not None:
        for layer in layers_to_ignore:
            ckpt.pop(layer)

    # Load the adjusted state_dict into the model instance.
    incompatible_keys = model.load_state_dict(ckpt, strict=False)

    # Log the incompatible keys during checkpoint loading.
    if len(incompatible_keys.missing_keys) > 0 or len(incompatible_keys.unexpected_keys) > 0:
        logger.info(f"Encountered incompatible keys during checkpoint loading. If intended, ignore.\n{incompatible_keys}")
    else:
        logger.info("Checkpoint loaded successfully.")

    return model


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator
    

import numpy as np
import cv2

def blend_3d_image_with_intensity_map(image: torch.Tensor, intensity_map: torch.Tensor, blend_factor: float) -> torch.Tensor:
    """
    Blend a 3D image with an intensity map displayed as a jet colormap overlay.

    Parameters:
    - image: 3D numpy array with shape (D, H, W) and values in range [0, 1]
    - intensity_map: 3D numpy array with shape (D, H, W) and values in range [0, 1]
    - blend_factor: float in range [0, 1] determining the blend weight

    Returns:
    - blended_image: 3D numpy array with shape (D, H, W, 3) and values in range [0, 1]
    """
    if image.shape != intensity_map.shape:
        raise ValueError("Image and intensity map must have the same shape")
    if not (0 <= blend_factor <= 1):
        raise ValueError("Blend factor must be in the range [0, 1]")
    
    image = image.numpy()
    intensity_map = intensity_map.numpy()

    image = image.squeeze()
    intensity_map = intensity_map.squeeze()

    # Normalize image to range [0, 255]
    image = (image * 255).astype(np.uint8)

    # Apply jet colormap to intensity map
    jet_colormap = plt.get_cmap('magma')
    intensity_map_colored = jet_colormap(intensity_map)[:, :, :, :3]  # Drop the alpha channel
    intensity_map_colored = (intensity_map_colored * 255).astype(np.uint8)

    # Blend the original image with the jet colormap overlay
    blended_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3), dtype=np.float32)
    for i in range(image.shape[0]):
        img_color = cv2.cvtColor(image[i], cv2.COLOR_GRAY2BGR)
        blended_image[i] = cv2.addWeighted(img_color.astype(np.float32), 1 - blend_factor, intensity_map_colored[i].astype(np.float32), blend_factor, 0)

    # Normalize blended image to range [0, 1]
    blended_image = blended_image / 255.0

    return torch.tensor(blended_image).unsqueeze(0)
