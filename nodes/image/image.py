import os
import torch

from ...core.node import node_path, node_prefix, main_cetegory
from ...core.image import *
from ...core.color import *

import folder_paths
from comfy_api.latest import io
from comfy_api.latest import ui

class ImageTransformNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaImageTransform",
            display_name=node_prefix() + " Artha Image Transform",
            category=main_cetegory() + "/Image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input(
                    "resize_mode",
                    options=["none", "resize", "crop", "pad"],
                    default="none",
                ),

                io.Combo.Input(
                    "upscale_method",
                    options=["nearest", "bilinear", "bicubic", "area", "lanczos"],
                    default="lanczos",
                ),

                io.Int.Input("target_width", default=512, min=64, max=4096, step=8),
                io.Int.Input("target_height", default=512, min=64, max=4096, step=8),

                io.Boolean.Input("keep_proportions", default=True),

                io.Combo.Input(
                    "rotation",
                    options=["0", "90", "180", "270", "custom"],
                    default="0",
                ),

                io.Float.Input(
                    "custom_angle",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=0.1,
                ),

                io.Boolean.Input("flip_horizontal", default=False),
                io.Boolean.Input("flip_vertical", default=False),
                io.Boolean.Input("normalize", default=True),
                io.Boolean.Input("auto_orient", default=False),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.Int.Output("width"),
                io.Int.Output("height"),
            ],
        )
    
    @classmethod
    def execute(self, image, mask=None, resize_mode="none", upscale_method="lanczos", target_width=512, target_height=512, keep_proportions=True, rotation="0", custom_angle=0.0, flip_horizontal=False, flip_vertical=False, normalize=False, auto_orient=False):
        
        img = tensor_to_pil(image)
        
        if rotation == "custom":
            
            img = rotate_image(img, custom_angle, expand=True)
        
        elif rotation != "0":
            
            img = rotate_image(img, int(rotation), expand=True)       

        if flip_horizontal or flip_vertical:
            
            img = flip_mirror_image(img, flip_horizontal, flip_vertical)
        
        if resize_mode == "resize":
            
            img = resize_image_proportional(img, target_width, target_height, keep_proportions, upscale_method)
        
        elif resize_mode == "crop":
            
            img = crop_image_to_aspect(img, target_width, target_height, upscale_method)
        
        elif resize_mode == "pad":
            
            img = pad_image_to_dimensions(img, target_width, target_height, upscale_method)
        
        image = pil_to_tensor(img, normalize)
        
        return (image, mask, img.width, img.height)
        
#################################################          
                    
class ImagePreviewNode(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaImagePreview",
            display_name=node_prefix() + " Artha Image Preview",
            category=main_cetegory() + "/Image",
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="Image",
                    optional=True,
                ),
            ],
            outputs=[], 
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        
        if images is None:
            
            return io.NodeOutput(None)

        return io.NodeOutput(
            None,
            ui=ui.PreviewImage(images, cls=cls)
        )

#################################################
        
class ImageSaveNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaImageSave",
            display_name=node_prefix() + " Artha Image Save",
            category=main_cetegory() + "/Image",
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="image",
                ),
                io.String.Input(
                    "filename_prefix",
                    display_name="prefix",
                    default="ComfyUI",
                    optional=True,
                ),
                io.Int.Input(
                    "compression",
                    display_name="compression",
                    default=4,
                    min=0,
                    max=9,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, images, compression, filename_prefix="ComfyUI") -> io.NodeOutput:

        if images is None:
            return io.NodeOutput(None)

        ui_result = ui.ImageSaveHelper.get_save_images_ui(
            images,
            filename_prefix=filename_prefix,
            cls=cls,
            compress_level=compression
        )

        return io.NodeOutput(
            None, 
            ui=ui_result 
        )

#################################################
        
class ImageColorMatchNode(io.ComfyNode):

    CATEGORY = main_cetegory() + "/Image"
    DESCRIPTION = "Advanced color matching between two images."

    COLOR_METHODS = [
        "reinhard",
        "mkl",
        "ciede2000",
        "lab_nn",
        "color_mood",
        "pdf",
    ]

    COLOR_METHODS_WITH_NONE = ["none"] + COLOR_METHODS

    MIXING_MODES = [
        "none",
        "blend",
        "sequential",
        "luminance",
        "detail",
    ]

    GAMUT_MODES = [
        "auto",
        "srgb",
        "adobe_rgb",
        "prophoto_rgb",
        "rec2020",
        "dci_p3",
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaImageColorMatch",
            display_name=node_prefix() + " Artha Color Match",
            category=cls.CATEGORY,
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input("source_image"),
                io.Image.Input("target_image"),

                io.Combo.Input(
                    "primary_method",
                    options=cls.COLOR_METHODS,
                ),

                io.Combo.Input(
                    "secondary_method",
                    options=cls.COLOR_METHODS_WITH_NONE,
                ),

                io.Combo.Input(
                    "mixing_mode",
                    options=cls.MIXING_MODES,
                ),

                io.Float.Input(
                    "primary_weight",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                ),

                io.Boolean.Input(
                    "preserve_luminance",
                    default=True,
                ),

                io.Int.Input(
                    "iterations",
                    default=1,
                    min=1,
                    max=10,
                ),

                io.Combo.Input(
                    "gamut_mode",
                    options=cls.GAMUT_MODES,
                    default="auto",
                ),

                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                ),

                io.Boolean.Input(
                    "use_regrain",
                    default=False,
                ),

                io.Float.Input(
                    "regrain_strength",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                ),

                io.Int.Input(
                    "regrain_radius",
                    default=3,
                    min=1,
                    max=15,
                ),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(
        cls,
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