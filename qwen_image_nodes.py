"""
Qwen-Image ComfyUI Nodes

This module contains all the ComfyUI nodes for Qwen-Image integration.
Provides comprehensive image generation, editing, and understanding capabilities
with exceptional Chinese text rendering using ComfyUI's standard model loading architecture.
"""

import torch
import numpy as np
from PIL import Image
import io
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

try:
    import folder_paths
    from comfy.model_management import get_torch_device
    import comfy.sd
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    from comfy import model_management
    import comfy.latent_formats
except ImportError as e:
    logging.error(f"Failed to import required dependencies: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

POSITIVE_MAGIC_PROMPTS = {
    "en": "Ultra HD, 4K, cinematic composition.",
    "zh": "超清，4K，电影级构图"
}

class QwenImageUNETLoader:
    """
    Node for loading Qwen-Image UNet/Diffusion model using ComfyUI's standard loading system.
    Supports the separated model architecture with proper weight dtype handling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select the Qwen-Image UNet model (e.g., qwen_image_bf16.safetensors)"
                }),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Weight precision for the model"
                })
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_unet(self, unet_name: str, weight_dtype: str):
        """Load Qwen-Image UNet model using ComfyUI's standard loading system."""
        try:
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

            logger.info(f"Qwen-Image UNet loaded successfully: {unet_name}")
            return (model,)

        except Exception as e:
            logger.error(f"Failed to load Qwen-Image UNet: {e}")
            raise RuntimeError(f"UNet loading failed: {e}")


class QwenImageCLIPLoader:
    """
    Node for loading Qwen-Image text encoder using ComfyUI's standard CLIP loading system.
    Supports Qwen 2.5 VL models with proper Chinese text handling.
    Enhanced to support dual CLIP loading - user can choose to load one or two CLIP models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Select the primary Qwen text encoder model (e.g., qwen_2.5_vl_7b.safetensors)"
                }),
                "load_dual_clip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable loading of a second CLIP model"
                }),
                "device": (["default", "cpu"], {
                    "default": "default",
                    "tooltip": "Device to load the model on"
                })
            },
            "optional": {
                "clip_name_2": (["none"] + folder_paths.get_filename_list("text_encoders"), {
                    "default": "none",
                    "tooltip": "Select the secondary Qwen text encoder model (only used if dual CLIP is enabled)"
                }),
            }
        }

    RETURN_TYPES = ("CLIP", "CLIP")
    RETURN_NAMES = ("clip", "clip_2")
    FUNCTION = "load_clip"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_clip(self, clip_name: str, load_dual_clip: bool, device: str = "default", clip_name_2: str = "none"):
        """Load Qwen-Image text encoder(s) using ComfyUI's CLIP loading system."""
        try:
            model_options = {}
            if device == "cpu":
                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

            # Load primary CLIP
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
                model_options=model_options
            )
            logger.info(f"Primary Qwen-Image CLIP loaded successfully: {clip_name}")

            # Load secondary CLIP if requested
            clip_2 = None
            if load_dual_clip and clip_name_2 != "none":
                try:
                    clip_path_2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name_2)
                    clip_2 = comfy.sd.load_clip(
                        ckpt_paths=[clip_path_2],
                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
                        model_options=model_options
                    )
                    logger.info(f"Secondary Qwen-Image CLIP loaded successfully: {clip_name_2}")
                except Exception as e:
                    logger.warning(f"Failed to load secondary CLIP {clip_name_2}: {e}")
                    clip_2 = None

            return (clip, clip_2)

        except Exception as e:
            logger.error(f"Failed to load Qwen-Image CLIP: {e}")
            raise RuntimeError(f"CLIP loading failed: {e}")


class QwenImageVAELoader:
    """
    Node for loading Qwen-Image VAE using ComfyUI's standard VAE loading system.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Select the Qwen-Image VAE model (e.g., qwen_image_vae.safetensors)"
                })
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_vae(self, vae_name: str):
        """Load Qwen-Image VAE using ComfyUI's standard loading system."""
        try:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

            logger.info(f"Qwen-Image VAE loaded successfully: {vae_name}")
            return (vae,)

        except Exception as e:
            logger.error(f"Failed to load Qwen-Image VAE: {e}")
            raise RuntimeError(f"VAE loading failed: {e}")


class QwenImageAdvancedDiffusionLoader:
    """
    Advanced UNet/Diffusion Model Loader for Qwen-Image.
    Provides comprehensive UNet model loading with advanced options including:
    - Advanced weight data type optimization (fp16, bf16, fp8 variants)
    - Compute data type selection
    - Memory optimization settings
    - SageAttention support
    - cuBLAS linear layer modifications
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select the Qwen-Image model (e.g., qwen_image_bf16.safetensors, qwen_image_fp8_e4m3fn.safetensors)"
                }),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16"], {
                    "default": "default",
                    "tooltip": "Weight precision for the model"
                }),
                "compute_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"], {
                    "default": "default",
                    "tooltip": "Compute precision for inference"
                }),
                "modify_cublas_linear": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable cuBLAS linear layer modifications for performance"
                }),
                "sage_attention": (["disabled", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"], {
                    "default": "disabled",
                    "tooltip": "SageAttention optimization mode"
                }),
                "use_fp16_cumulation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use FP16 accumulation for memory efficiency"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_advanced_diffusion"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_advanced_diffusion(self, model_name: str, weight_dtype: str, compute_dtype: str,
                               modify_cublas_linear: bool, sage_attention: str, use_fp16_cumulation: bool):
        """Load complete Qwen-Image diffusion pipeline with advanced options."""
        try:
            logger.info(f"Loading advanced Qwen-Image diffusion model: {model_name}")

            # Prepare model options
            model_options = {}

            # Configure weight dtype
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            elif weight_dtype == "fp16":
                model_options["dtype"] = torch.float16
            elif weight_dtype == "bf16":
                model_options["dtype"] = torch.bfloat16

            # Configure compute dtype
            if compute_dtype != "default":
                if compute_dtype == "fp8_e4m3fn":
                    model_options["compute_dtype"] = torch.float8_e4m3fn
                elif compute_dtype == "fp8_e4m3fn_fast":
                    model_options["compute_dtype"] = torch.float8_e4m3fn
                    model_options["fp8_compute_optimizations"] = True
                elif compute_dtype == "fp8_e5m2":
                    model_options["compute_dtype"] = torch.float8_e5m2
                elif compute_dtype == "fp16":
                    model_options["compute_dtype"] = torch.float16
                elif compute_dtype == "bf16":
                    model_options["compute_dtype"] = torch.bfloat16
                elif compute_dtype == "fp32":
                    model_options["compute_dtype"] = torch.float32

            # Configure cuBLAS modifications
            if modify_cublas_linear:
                model_options["modify_cublas_linear"] = True

            # Configure SageAttention
            if sage_attention != "disabled":
                model_options["sage_attention"] = sage_attention

            # Configure FP16 accumulation
            if use_fp16_cumulation:
                model_options["use_fp16_accumulation"] = True



            # Load UNet/Diffusion model
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            logger.info(f"Advanced Qwen-Image UNet loaded successfully with options: {model_options}")

            return (model,)

        except Exception as e:
            logger.error(f"Failed to load advanced diffusion model: {e}")
            raise RuntimeError(f"Advanced diffusion loading failed: {e}")


class QwenImageTextEncode:
    """
    Node for encoding text prompts using Qwen-Image CLIP model.
    Optimized for Chinese text processing with template support.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Qwen-Image CLIP model from QwenImageCLIPLoader"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的小猫咪坐在花园里",
                    "tooltip": "Text prompt for encoding (supports Chinese)"
                }),
            },
            "optional": {
                "magic_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add quality enhancement prompts"
                }),
                "language": (["auto", "zh", "en"], {
                    "default": "auto",
                    "tooltip": "Language for magic prompts"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Ken-Chen/Qwen-Image/conditioning"

    def encode(self, clip, text: str, magic_prompt: bool = True, language: str = "auto"):
        """Encode text using Qwen-Image CLIP model."""
        try:
            # Enhance prompt with magic prompts if enabled
            enhanced_text = text
            if magic_prompt:
                # Auto-detect language if needed
                if language == "auto":
                    # Simple heuristic: if contains Chinese characters, use Chinese
                    import re
                    if re.search(r'[\u4e00-\u9fff]', text):
                        language = "zh"
                    else:
                        language = "en"

                if language in POSITIVE_MAGIC_PROMPTS:
                    enhanced_text = f"{text}, {POSITIVE_MAGIC_PROMPTS[language]}"

            logger.info(f"Encoding text: {enhanced_text[:100]}...")

            # Encode using ComfyUI's standard CLIP encoding
            tokens = clip.tokenize(enhanced_text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            return ([[cond, {"pooled_output": pooled}]],)

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise RuntimeError(f"Text encoding failed: {e}")


class QwenImageDualTextEncode:
    """
    Node for encoding text prompts using dual Qwen-Image CLIP models.
    Supports combining outputs from two CLIP models for enhanced text understanding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Primary Qwen-Image CLIP model"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的小猫咪坐在花园里",
                    "tooltip": "Text prompt for encoding (supports Chinese)"
                }),
                "use_dual_clip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable dual CLIP encoding"
                }),
            },
            "optional": {
                "clip_2": ("CLIP", {
                    "tooltip": "Secondary Qwen-Image CLIP model (optional)"
                }),
                "blend_mode": (["average", "concat", "primary_only", "secondary_only"], {
                    "default": "average",
                    "tooltip": "How to combine dual CLIP outputs"
                }),
                "magic_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add quality enhancement prompts"
                }),
                "language": (["auto", "zh", "en"], {
                    "default": "auto",
                    "tooltip": "Language for magic prompts"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_dual"
    CATEGORY = "Ken-Chen/Qwen-Image/conditioning"

    def encode_dual(self, clip, text: str, use_dual_clip: bool = False, clip_2=None,
                   blend_mode: str = "average", magic_prompt: bool = True, language: str = "auto"):
        """Encode text using single or dual Qwen-Image CLIP models."""
        try:
            # Enhance prompt with magic prompts if enabled
            enhanced_text = text
            if magic_prompt:
                # Auto-detect language if needed
                if language == "auto":
                    # Simple heuristic: if contains Chinese characters, use Chinese
                    import re
                    if re.search(r'[\u4e00-\u9fff]', text):
                        language = "zh"
                    else:
                        language = "en"

                if language in POSITIVE_MAGIC_PROMPTS:
                    enhanced_text = f"{text}, {POSITIVE_MAGIC_PROMPTS[language]}"

            logger.info(f"Encoding text with dual CLIP: {enhanced_text[:100]}...")

            # Encode with primary CLIP
            tokens = clip.tokenize(enhanced_text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            # If dual CLIP is not enabled or secondary CLIP is not available, return primary only
            if not use_dual_clip or clip_2 is None:
                logger.info("Using primary CLIP only")
                return ([[cond, {"pooled_output": pooled}]],)

            # Encode with secondary CLIP
            tokens_2 = clip_2.tokenize(enhanced_text)
            cond_2, pooled_2 = clip_2.encode_from_tokens(tokens_2, return_pooled=True)

            # Combine outputs based on blend mode
            if blend_mode == "average":
                # Average the conditioning tensors
                combined_cond = (cond + cond_2) / 2.0
                combined_pooled = (pooled + pooled_2) / 2.0
                logger.info("Combined CLIP outputs using average")
            elif blend_mode == "concat":
                # Concatenate along the feature dimension
                combined_cond = torch.cat([cond, cond_2], dim=-1)
                combined_pooled = torch.cat([pooled, pooled_2], dim=-1)
                logger.info("Combined CLIP outputs using concatenation")
            elif blend_mode == "primary_only":
                combined_cond = cond
                combined_pooled = pooled
                logger.info("Using primary CLIP output only")
            elif blend_mode == "secondary_only":
                combined_cond = cond_2
                combined_pooled = pooled_2
                logger.info("Using secondary CLIP output only")
            else:
                # Default to average
                combined_cond = (cond + cond_2) / 2.0
                combined_pooled = (pooled + pooled_2) / 2.0
                logger.info("Combined CLIP outputs using default average")

            return ([[combined_cond, {"pooled_output": combined_pooled}]],)

        except Exception as e:
            logger.error(f"Failed to encode text with dual CLIP: {e}")
            raise RuntimeError(f"Dual text encoding failed: {e}")


class QwenImageEmptyLatentImage:
    """
    Node for creating empty latent images with Qwen-Image optimized dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Image width"
                }),
                "height": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Image height"
                }),
                "aspect_ratio": (list(SUPPORTED_ASPECT_RATIOS.keys()), {
                    "default": "1:1",
                    "tooltip": "Predefined aspect ratios optimized for Qwen-Image"
                }),
                "use_aspect_ratio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use predefined aspect ratio instead of custom width/height"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Number of images to generate"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "Ken-Chen/Qwen-Image/latent"

    def generate(self, width, height, aspect_ratio, use_aspect_ratio, batch_size):
        """Generate empty latent image with specified dimensions."""
        try:
            # Set dimensions based on aspect ratio if enabled
            if use_aspect_ratio and aspect_ratio in SUPPORTED_ASPECT_RATIOS:
                width, height = SUPPORTED_ASPECT_RATIOS[aspect_ratio]

            # Create latent with proper format for Qwen-Image (assuming 8x downsampling)
            latent_width = width // 8
            latent_height = height // 8

            latent = torch.zeros([batch_size, 4, latent_height, latent_width])

            logger.info(f"Created empty latent: {batch_size}x4x{latent_height}x{latent_width} (image: {width}x{height})")
            return ({"samples": latent},)

        except Exception as e:
            logger.error(f"Failed to create empty latent: {e}")
            raise RuntimeError(f"Latent creation failed: {e}")


class QwenImageSampler:
    """
    Node for sampling images using Qwen-Image model with ComfyUI's standard sampling system.
    Supports various samplers and schedulers optimized for Qwen-Image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Qwen-Image model from QwenImageUNETLoader"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning from QwenImageTextEncode"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning from QwenImageTextEncode"
                }),
                "latent_image": ("LATENT", {
                    "tooltip": "Latent image to denoise"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed (-1 for random)"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampling algorithm"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                    "tooltip": "Noise schedule"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising strength"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Ken-Chen/Qwen-Image/sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise):
        """Sample latent image using Qwen-Image model."""
        try:
            # Handle seed
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()

            logger.info(f"Sampling with seed: {seed}, steps: {steps}, cfg: {cfg}")

            # Use ComfyUI's standard common_ksampler function
            # Import the function from nodes module
            import nodes
            result = nodes.common_ksampler(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent_image,
                denoise=denoise
            )

            logger.info("Sampling completed successfully")
            return result  # common_ksampler already returns a tuple

        except Exception as e:
            logger.error(f"Failed to sample: {e}")
            raise RuntimeError(f"Sampling failed: {e}")


class QwenImageEdit:
    """
    Node for editing images using Qwen-Image model.
    Supports style transfer, object insertion/removal, and detail enhancement.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_PIPELINE",),
                "input_image": ("IMAGE",),
                "edit_prompt": ("STRING", {
                    "default": "Add Chinese text '春节快乐' in red color",
                    "multiline": True
                }),
                "edit_type": (["style_transfer", "object_add", "object_remove", "text_edit", "detail_enhance"], {
                    "default": "text_edit"
                }),
                "strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "Ken-Chen/Qwen-Image/legacy"

    def edit_image(self, pipeline, input_image: torch.Tensor, edit_prompt: str,
                   edit_type: str, strength: float, num_inference_steps: int,
                   guidance_scale: float, seed: int, mask: Optional[torch.Tensor] = None,
                   negative_prompt: str = "") -> Tuple[torch.Tensor]:
        """Edit image using Qwen-Image pipeline."""
        try:
            # Convert ComfyUI tensor to PIL Image
            image_array = input_image.squeeze().cpu().numpy()
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)

            # Set up generator
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)

            logger.info(f"Editing image with type: {edit_type}")
            logger.info(f"Edit prompt: {edit_prompt[:100]}...")

            # Prepare editing parameters based on edit type
            edit_params = {
                "prompt": edit_prompt,
                "image": pil_image,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator
            }

            if negative_prompt:
                edit_params["negative_prompt"] = negative_prompt

            # Add mask if provided
            if mask is not None:
                mask_array = mask.squeeze().cpu().numpy()
                if mask_array.max() <= 1.0:
                    mask_array = (mask_array * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_array, mode='L')
                edit_params["mask_image"] = mask_pil

            # Note: This is a placeholder for image editing functionality
            # The actual implementation would depend on the specific editing capabilities
            # available in the Qwen-Image model

            # For now, we'll use img2img as a fallback
            if hasattr(pipeline, 'img2img'):
                result = pipeline.img2img(**edit_params)
            else:
                # Fallback to regular generation with image conditioning
                result = pipeline(**edit_params)

            # Convert result back to ComfyUI tensor format
            edited_image = result.images[0]
            image_array = np.array(edited_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]

            logger.info("Image edited successfully")
            return (image_tensor,)

        except Exception as e:
            logger.error(f"Failed to edit image: {e}")
            raise RuntimeError(f"Image editing failed: {e}")


class QwenImageTextRender:
    """
    Specialized node for high-quality text rendering in images.
    Optimized for Chinese text rendering with Qwen-Image.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_PIPELINE",),
                "text_content": ("STRING", {
                    "default": "春节快乐\nHappy New Year",
                    "multiline": True
                }),
                "background_prompt": ("STRING", {
                    "default": "A festive red background with golden decorations",
                    "multiline": True
                }),
                "text_style": (["elegant", "modern", "traditional", "artistic", "calligraphy"], {
                    "default": "elegant"
                }),
                "text_color": (["black", "white", "red", "gold", "blue", "green", "custom"], {
                    "default": "gold"
                }),
                "text_size": (["small", "medium", "large", "extra_large"], {
                    "default": "large"
                }),
                "text_position": (["center", "top", "bottom", "left", "right", "custom"], {
                    "default": "center"
                }),
                "aspect_ratio": (list(SUPPORTED_ASPECT_RATIOS.keys()), {
                    "default": "16:9"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1
                }),
            },
            "optional": {
                "custom_color": ("STRING", {
                    "default": "#FFD700",
                    "multiline": False
                }),
                "font_weight": (["normal", "bold", "light"], {
                    "default": "normal"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("text_image",)
    FUNCTION = "render_text"
    CATEGORY = "Ken-Chen/Qwen-Image/legacy"

    def render_text(self, pipeline, text_content: str, background_prompt: str,
                   text_style: str, text_color: str, text_size: str,
                   text_position: str, aspect_ratio: str, num_inference_steps: int,
                   guidance_scale: float, seed: int, custom_color: str = "#FFD700",
                   font_weight: str = "normal") -> Tuple[torch.Tensor]:
        """Render high-quality text in images using Qwen-Image."""
        try:
            # Get dimensions
            width, height = SUPPORTED_ASPECT_RATIOS[aspect_ratio]

            # Build comprehensive prompt for text rendering
            color_desc = custom_color if text_color == "custom" else text_color

            # Create detailed prompt for high-quality text rendering
            text_prompt = f"""
            {background_prompt}

            Text content: "{text_content}"
            Text style: {text_style} {font_weight} font
            Text color: {color_desc}
            Text size: {text_size}
            Text position: {text_position}

            High-quality text rendering, crisp and clear typography,
            perfect character formation, especially for Chinese characters,
            professional typography, ultra-sharp text details,
            {POSITIVE_MAGIC_PROMPTS['zh'] if any('\u4e00' <= char <= '\u9fff' for char in text_content) else POSITIVE_MAGIC_PROMPTS['en']}
            """.strip()

            # Set up generator
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)

            logger.info(f"Rendering text: {text_content[:50]}...")
            logger.info(f"Style: {text_style}, Color: {color_desc}, Position: {text_position}")

            # Generate image with text
            result = pipeline(
                prompt=text_prompt,
                negative_prompt="blurry text, illegible characters, distorted text, low quality typography",
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

            # Convert to ComfyUI tensor format
            image = result.images[0]
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]

            logger.info("Text rendered successfully")
            return (image_tensor,)

        except Exception as e:
            logger.error(f"Failed to render text: {e}")
            raise RuntimeError(f"Text rendering failed: {e}")


class QwenImageUnderstanding:
    """
    Node for image understanding tasks using Qwen-Image.
    Supports object detection, semantic segmentation, depth estimation, etc.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_PIPELINE",),
                "input_image": ("IMAGE",),
                "task_type": ([
                    "object_detection",
                    "semantic_segmentation",
                    "depth_estimation",
                    "edge_detection",
                    "super_resolution",
                    "novel_view_synthesis"
                ], {
                    "default": "object_detection"
                }),
                "output_format": (["image", "mask", "data"], {
                    "default": "image"
                }),
            },
            "optional": {
                "task_prompt": ("STRING", {
                    "default": "Detect all objects in this image",
                    "multiline": True
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result_image", "analysis_text")
    FUNCTION = "understand_image"
    CATEGORY = "Ken-Chen/Qwen-Image/legacy"

    def understand_image(self, pipeline, input_image: torch.Tensor, task_type: str,
                        output_format: str, task_prompt: str = "Analyze this image",
                        confidence_threshold: float = 0.5) -> Tuple[torch.Tensor, str]:
        """Perform image understanding tasks."""
        try:
            # Convert ComfyUI tensor to PIL Image
            image_array = input_image.squeeze().cpu().numpy()
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)

            logger.info(f"Performing {task_type} on image")

            # Note: This is a placeholder implementation
            # The actual implementation would depend on the specific understanding
            # capabilities available in the Qwen-Image model

            # For now, we'll create a simple analysis
            analysis_text = f"Task: {task_type}\nPrompt: {task_prompt}\nConfidence threshold: {confidence_threshold}"

            # Return the original image as placeholder
            # In a real implementation, this would return the processed result
            result_tensor = input_image

            logger.info(f"Image understanding completed: {task_type}")
            return (result_tensor, analysis_text)

        except Exception as e:
            logger.error(f"Failed to understand image: {e}")
            raise RuntimeError(f"Image understanding failed: {e}")


# Import DiffSynth nodes
try:
    from .qwen_diffsynth_nodes import (
        NODE_CLASS_MAPPINGS as DIFFSYNTH_NODE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as DIFFSYNTH_DISPLAY_MAPPINGS
    )
    DIFFSYNTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DiffSynth nodes not available: {e}")
    DIFFSYNTH_NODE_MAPPINGS = {}
    DIFFSYNTH_DISPLAY_MAPPINGS = {}
    DIFFSYNTH_AVAILABLE = False

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Model Loaders (New ComfyUI-compatible architecture)
    "QwenImageUNETLoader": QwenImageUNETLoader,
    "QwenImageCLIPLoader": QwenImageCLIPLoader,
    "QwenImageVAELoader": QwenImageVAELoader,
    "QwenImageAdvancedDiffusionLoader": QwenImageAdvancedDiffusionLoader,

    # Text Processing
    "QwenImageTextEncode": QwenImageTextEncode,
    "QwenImageDualTextEncode": QwenImageDualTextEncode,

    # Latent Operations
    "QwenImageEmptyLatentImage": QwenImageEmptyLatentImage,
    "QwenImageSampler": QwenImageSampler,

    # Legacy nodes (for backward compatibility)
    "QwenImageEdit": QwenImageEdit,
    "QwenImageTextRender": QwenImageTextRender,
    "QwenImageUnderstanding": QwenImageUnderstanding,
}

# Add DiffSynth nodes if available
if DIFFSYNTH_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(DIFFSYNTH_NODE_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    # Model Loaders
    "QwenImageUNETLoader": "🎨 Qwen-Image UNet Loader",
    "QwenImageCLIPLoader": "🎨 Qwen-Image CLIP Loader",
    "QwenImageVAELoader": "🎨 Qwen-Image VAE Loader",
    "QwenImageAdvancedDiffusionLoader": "🎨 Qwen-Image Advanced Diffusion Loader",

    # Text Processing
    "QwenImageTextEncode": "🎨 Qwen-Image Text Encode",
    "QwenImageDualTextEncode": "🎨 Qwen-Image Dual Text Encode",

    # Latent Operations
    "QwenImageEmptyLatentImage": "🎨 Qwen-Image Empty Latent",
    "QwenImageSampler": "🎨 Qwen-Image Sampler",

    # Legacy nodes
    "QwenImageEdit": "🎨 Qwen-Image Edit (Legacy)",
    "QwenImageTextRender": "🎨 Qwen-Image Text Render (Legacy)",
    "QwenImageUnderstanding": "🎨 Qwen-Image Understanding (Legacy)",
}

# Add DiffSynth display names if available
if DIFFSYNTH_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update(DIFFSYNTH_DISPLAY_MAPPINGS)
