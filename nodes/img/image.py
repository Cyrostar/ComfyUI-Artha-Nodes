import os
import torch
import folder_paths
from ...core.node import node_path, node_prefix, main_cetegory
from ...core.image import *
from ...core.color import *

class ImageLoader:

    CATEGORY = main_cetegory() + "/IMG"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get list of supported image formats
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]
        
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "resize_mode": (["none", "resize", "crop", "pad"], {"default": "none"}),
                "upscale_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"], {"default": "lanczos"}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "keep_proportions": ("BOOLEAN", {"default": True}),
                "rotation": (["0", "90", "180", "270", "custom"], {"default": "0"}),
                "custom_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                "normalize": ("BOOLEAN", {"default": True}),
                "auto_orient": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "artha_main"

    
    def artha_main(self, image, resize_mode="none", upscale_method="lanczos", target_width=512, target_height=512, keep_proportions=True, rotation="0", custom_angle=0.0, flip_horizontal=False, flip_vertical=False, normalize=False, auto_orient=False):
        
        # Get the full path to the image
        image_path = folder_paths.get_annotated_filepath(image)
        
        try:
            # Load the image using utility function
            img = load_pil_image(image_path, auto_orient)
            
            # Apply rotation using utility function
            if rotation == "custom":
                img = rotate_image(img, custom_angle, expand=True)
            elif rotation != "0":
                img = rotate_image(img, int(rotation), expand=True)
            
            # Apply flips/mirrors using utility function
            if flip_horizontal or flip_vertical:
                img = flip_mirror_image(img, flip_horizontal, flip_vertical)
            
            # Handle resizing based on mode using utility functions
            if resize_mode == "resize":
                img = resize_image_proportional(img, target_width, target_height, 
                                              keep_proportions, upscale_method)
            
            elif resize_mode == "crop":
                img = crop_image_to_aspect(img, target_width, target_height, upscale_method)
            
            elif resize_mode == "pad":
                img = pad_image_to_dimensions(img, target_width, target_height, upscale_method)
            
            # Convert to tensor using utility function
            image_tensor = pil_to_tensor(img, normalize)
            
            # Create mask using utility function
            mask = create_mask_from_image(img)
            
            return (image_tensor, mask, img.width, img.height)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return fallback tensors using utility function
            fallback_img, fallback_mask = create_fallback_tensors(target_width, target_height, normalize)
            return (fallback_img, fallback_mask, target_width, target_height)
            
class ImageDisplay:

    CATEGORY = main_cetegory() + "/IMG"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ([1024, 2048, 4096], {"default": 2048}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "artha_main"
    OUTPUT_NODE = True 

    @classmethod
    def artha_main(cls, image, resolution):
        
        encoded = image_display(image, max_size=resolution)
                   
        return {
        "ui": {"image": [encoded]}, 
        "result": (image,)
        }
        
class ColorMatch:
    
    CATEGORY = main_cetegory() + "/IMG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "primary_method": (["reinhard", "mkl", "ciede2000", "lab_nn", "color_mood", "pdf"],),
                "secondary_method": (["none", "reinhard", "mkl", "ciede2000", "lab_nn", "color_mood", "pdf"],),
                "mixing_mode": (["none", "blend", "sequential", "luminance", "detail"],),
                "primary_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "preserve_luminance": ("BOOLEAN", {"default": True}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
                "gamut_mode": (["auto", "srgb", "adobe_rgb", "prophoto_rgb", "rec2020", "dci_p3"], {"default": "auto"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "use_regrain": ("BOOLEAN", {"default": False}),
                "regrain_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0}),
                "regrain_radius": ("INT", {"default": 3, "min": 1, "max": 15}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "artha_main"

    # ------------------------------------------
    # Main function
    # ------------------------------------------
    def artha_main(
        self,
        source_image,
        target_image,
        primary_method,
        secondary_method,
        mixing_mode,
        primary_weight,
        preserve_luminance,
        iterations,
        gamut_mode,
        strength,
        use_regrain,
        regrain_strength,
        regrain_radius
    ):
        # Run primary transfer
        result = apply_single_method(source_image, target_image, primary_method, gamut_mode)

        # Iterative refinement
        for _ in range(iterations - 1):
            result = apply_single_method(result, target_image, primary_method, gamut_mode)

        # Apply secondary method if requested
        if secondary_method != "none":
            secondary_result = apply_single_method(result, target_image, secondary_method, gamut_mode)
            if mixing_mode == "blend":
                result = (primary_weight * result + (1 - primary_weight) * secondary_result)
            elif mixing_mode == "sequential":
                result = secondary_result
            elif mixing_mode == "luminance":
                result = luminance_split_mixing(result, secondary_result, lambda s, t: secondary_result)
            elif mixing_mode == "detail":
                result = detail_preserve_mixing(result, secondary_result, lambda s, t: secondary_result)

        # Luminance preservation toggle
        if preserve_luminance:
            result = luminance_split_mixing(source_image[0].cpu().numpy().astype("float32"), result, lambda s, t: result)

        # Regrain (restore fine details from source)
        if use_regrain:
            result = regrain(result, source_image[0].cpu().numpy().astype("float32"), strength=regrain_strength, radius=regrain_radius)

        # Apply strength factor (blend with original source)
        source_np = source_image[0].cpu().numpy().astype("float32")
        result = result * strength + (1 - strength) * source_np
        result = torch.from_numpy(result).unsqueeze(0).clamp(0, 1)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "Image Loader": ImageLoader,
    "Image Display": ImageDisplay,
    "Color Match": ColorMatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Loader": node_prefix() + " IMAGE LOADER",
    "Image Display": node_prefix() + " IMAGE DISPLAY",
    "Color Match": node_prefix() + " COLOR MATCH",
}