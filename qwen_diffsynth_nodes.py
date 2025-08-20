"""
Qwen-Image DiffSynth Integration Nodes
Based on DiffSynth-Studio implementation with advanced memory management and ControlNet/LoRA support.
"""

import os
import torch
import folder_paths
import comfy.model_management as mm
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Device management
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
torch_dtype = torch.bfloat16

# Optional mmgp offload for VRAM profiling (from QwenImage-Diffsynth_8)
try:
    from mmgp import offload as mmgp_offload
    from mmgp.offload import profile_type as mmgp_profile_type
    MMGP_AVAILABLE = True
    logger.info("mmgp.offload is available for VRAM profiling")
except Exception as _e:
    MMGP_AVAILABLE = False
    logger.warning(f"mmgp.offload not available: {_e}")

# VRAM optimization options
VRAM_OPTIMIZATION_OPTS = [
    'No_Optimization',
    'HighRAM_HighVRAM',
    'HighRAM_LowVRAM',
    'LowRAM_HighVRAM',
    'LowRAM_LowVRAM',
    'VerylowRAM_LowVRAM',
]

# ---- Text prompt normalization (remove leading "字："/"text:" and surrounding quotes) ----
def _sanitize_text_prompt(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text.strip()

    # Remove common prefixes like "字："/"字:"/"文字："/"文本："/"text:" (fullwidth or halfwidth colon)
    prefixes = [
        "字", "文字", "文本", "text", "Text", "TEXT"
    ]
    for p in prefixes:
        for colon in (":", "："):
            pref = f"{p}{colon}"
            if s.startswith(pref):
                s = s[len(pref):].strip()
                break
    # Also handle patterns like: 字 “内容” / 字: “内容” with optional spaces
    for p in prefixes:
        for colon in (":", "：", ""):
            pref = f"{p}{colon}"
            if s.startswith(pref):
                s = s[len(pref):].lstrip()
                break

    # Strip wrapping quotes/brackets if they match as a pair
    pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
        "「": "」",
        "『": "』",
        "《": "》",
        "(": ")",
        "（": "）",
        "[": "]",
        "【": "】",
        "{": "}",
    }
    changed = True
    while changed and len(s) >= 2:
        changed = False
        for a, b in pairs.items():
            if s.startswith(a) and s.endswith(b):
                inner = s[len(a):-len(b)].strip()
                if inner:
                    s = inner
                    changed = True
                    break
    return s.strip()

# A permissive type that matches any connection in ComfyUI type checks
class AnyType(str):
    def __eq__(self, _):
        return True
    def __ne__(self, _):
        return False

ANY = AnyType("*")


# Lazy DiffSynth import to avoid side effects during module import
# Returns: (available, QwenImagePipeline, ModelConfig, ControlNetInput, ModelManager)
def _get_diffsynth_classes():
    try:
        # Import all classes from the same module to ensure consistency
        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, ControlNetInput
        from diffsynth.models.model_manager import ModelManager
        return True, QwenImagePipeline, ModelConfig, ControlNetInput, ModelManager
    except Exception as e:
        logger.warning(f"DiffSynth-Studio not available or failed to import safely: {e}")

        class ModelConfig:
            def __init__(self, model_id=None, origin_file_pattern=None, **kwargs):
                self.model_id = model_id
                self.origin_file_pattern = origin_file_pattern
                self.__dict__.update(kwargs)

        class ControlNetInput:
            def __init__(self, image=None, **kwargs):
                self.image = image
                self.__dict__.update(kwargs)

        class QwenImagePipeline:
            @classmethod
            def from_pretrained(cls, torch_dtype=None, device=None, model_configs=None, tokenizer_config=None, **kwargs):
                return None

        return False, QwenImagePipeline, ModelConfig, ControlNetInput, None

# Utility functions
def pil2comfy(pil_image):
    """Convert PIL Image to ComfyUI tensor format."""
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def comfy2pil(comfy_tensor):
    """Convert ComfyUI tensor to PIL Image."""
    i = 255. * comfy_tensor.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


# Helper to treat various 'none' selections from saved workflows/UI
def _is_none_choice(val):
    try:
        if val is None:
            return True
        s = str(val).strip()
        return s == "" or s.lower() in ("none", "null")
    except Exception:
        return False


class QwenPipelineWrapper:
    """
    Wrapper class for DiffSynth pipeline to provide consistent interface.
    """
    def __init__(self, pipe, model_manager, config):
        self.pipe = pipe
        self.model_manager = model_manager
        self.config = config
        self.is_mock = config.get('mock', False)

    def __call__(self, **kwargs):
        """Call the wrapped pipeline with proper argument handling."""
        if self.is_mock or self.pipe is None:
            # Mock implementation for testing
            return self._mock_generation(**kwargs)

        try:
            # Call the actual DiffSynth pipeline
            result = self.pipe(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Pipeline call failed: {e}")
            return self._mock_generation(**kwargs)

    def _mock_generation(self, **kwargs):
        """Generate a mock image for testing purposes."""
        width = kwargs.get('width', 1024)
        height = kwargs.get('height', 1024)
        prompt = kwargs.get('prompt', 'test image')

        # Create a simple test image
        import numpy as np
        from PIL import Image

        # Create gradient background
        image_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a colorful gradient
        for y in range(height):
            for x in range(width):
                image_array[y, x, 0] = int((x / width) * 255)  # Red gradient
                image_array[y, x, 1] = int((y / height) * 255)  # Green gradient
                image_array[y, x, 2] = int(((x + y) / (width + height)) * 255)  # Blue gradient

        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)

        # Return in expected format
        class MockResult:
            def __init__(self, image):
                self.images = [image]

        return MockResult(pil_image)

class QwenImageDiffSynthLoRALoader:
    """
    Load LoRA models for Qwen-Image DiffSynth pipeline.
    Supports single LoRA loading with weight control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Select LoRA model file"
                }),
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "LoRA weight multiplier"
                })
            },
            "optional": {
                # Allow chaining: feed an existing QWEN_LORA to append more LoRAs
                "lora": ("QWEN_LORA", {
                    "tooltip": "Chain another LoRA config (to load multiple LoRAs sequentially)"
                })
            }
        }

    RETURN_TYPES = ("QWEN_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_lora(self, lora_name: str, weight: float, lora: Optional[Dict] = None):
        """Load a LoRA model; if an input lora config is provided, append to it for chaining."""
        try:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

            new_config = {
                "lora_path": [lora_path],
                "lora_weight": [weight],
                "lora_name": [lora_name]
            }

            # If an upstream LoRA config is provided, append to it so users can chain multiple LoRAs
            if lora and isinstance(lora, Dict) or isinstance(lora, dict):
                try:
                    def to_list(x, default=None):
                        if x is None:
                            return [] if default is None else [default]
                        return x if isinstance(x, list) else [x]

                    existing_paths = to_list(lora.get("lora_path"))
                    existing_weights = to_list(lora.get("lora_weight"), 1.0)
                    existing_names = to_list(lora.get("lora_name"), "unknown")

                    combined_config = {
                        "lora_path": existing_paths + new_config["lora_path"],
                        "lora_weight": existing_weights + new_config["lora_weight"],
                        "lora_name": existing_names + new_config["lora_name"],
                    }
                    logger.info(f"LoRA chained: {lora_name} with weight {weight}; total={len(combined_config['lora_path'])}")
                    return (combined_config,)
                except Exception:
                    # Fallback to just the new one if incoming config is malformed
                    logger.warning("Incoming QWEN_LORA malformed; returning only the newly loaded LoRA")
                    logger.info(f"LoRA loaded: {lora_name} with weight {weight}")
                    return (new_config,)
            else:
                logger.info(f"LoRA loaded: {lora_name} with weight {weight}")
                return (new_config,)

        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_name}: {e}")
            raise RuntimeError(f"LoRA loading failed: {e}")


class QwenImageDiffSynthLoRAMulti:
    """
    Combine multiple LoRA models for Qwen-Image DiffSynth pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_a": ("QWEN_LORA", {
                    "tooltip": "First LoRA configuration"
                }),
                "lora_b": ("QWEN_LORA", {
                    "tooltip": "Second LoRA configuration"
                })
            }
        }

    RETURN_TYPES = ("QWEN_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "combine_loras"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def combine_loras(self, lora_a: Dict, lora_b: Dict):
        """Combine multiple LoRA configurations."""
        try:
            combined_config = {
                "lora_path": lora_a["lora_path"] + lora_b["lora_path"],
                "lora_weight": lora_a["lora_weight"] + lora_b["lora_weight"],
                "lora_name": lora_a["lora_name"] + lora_b["lora_name"]
            }

            logger.info(f"Combined LoRAs: {combined_config['lora_name']}")
            return (combined_config,)

        except Exception as e:
            logger.error(f"Failed to combine LoRAs: {e}")
            raise RuntimeError(f"LoRA combination failed: {e}")


class QwenImageDiffSynthControlNetLoader:
    """
    Load ControlNet models for Qwen-Image DiffSynth pipeline.
    Supports Canny, Depth, and other ControlNet types.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"), {
                    "tooltip": "Select ControlNet model file"
                }),
                "controlnet_type": (["canny", "depth", "pose", "normal", "seg", "inpaint"], {
                    "default": "canny",
                    "tooltip": "Type of ControlNet"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "ControlNet strength"
                })
            }
        }

    RETURN_TYPES = ("QWEN_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_controlnet(self, controlnet_name: str, controlnet_type: str, strength: float):
        """Load ControlNet model."""
        try:
            controlnet_path = folder_paths.get_full_path_or_raise("controlnet", controlnet_name)

            controlnet_config = {
                "controlnet_path": controlnet_path,
                "controlnet_type": controlnet_type,
                "controlnet_strength": strength,
                "controlnet_name": controlnet_name
            }

            logger.info(f"ControlNet loaded: {controlnet_name} ({controlnet_type}) with strength {strength}")
            return (controlnet_config,)

        except Exception as e:
            logger.error(f"Failed to load ControlNet {controlnet_name}: {e}")
            raise RuntimeError(f"ControlNet loading failed: {e}")


class QwenImageDiffSynthPipelineLoader:
    """
    Advanced Qwen-Image pipeline loader with intelligent memory management.
    Ken-Chen's optimized pipeline with LoRA and ControlNet integration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_optimization": (VRAM_OPTIMIZATION_OPTS, {
                    "default": "HighRAM_LowVRAM",
                    "tooltip": "VRAM optimization strategy"
                }),
                "offload_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload models to CPU when not in use"
                }),
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision (now supports FP8 e4m3fn/e5m2 if your PyTorch build and GPU support it)"
                }),
                "base_model": (["auto", "Qwen-Image", "Qwen-Image-EliGen", "Qwen-Image-Distill-Full"], {
                    "default": "auto",
                    "tooltip": "选择基础模型：自动/原版/带EliGen/蒸馏版"
                }),
            },
            "optional": {
                # 保留原链路输入，仍可用 LoRA Loader 节点传入
                "lora": ("QWEN_LORA", {
                    "tooltip": "LoRA configuration (optional)"
                }),
                "controlnet": ("QWEN_CONTROLNET", {
                    "tooltip": "ControlNet configuration (optional)"
                }),
                # 直接从 loras 目录选择（与 lightning_lora 一致体验）
                "lightning_lora": (["none", "None", "NONE", "null", "Null", "NULL"] + folder_paths.get_filename_list("loras"), {
                    "default": "none",
                    "tooltip": "Lightning LoRA for faster inference"
                }),
                "lora_file": (["none", "None", "NONE", "null", "Null", "NULL"] + folder_paths.get_filename_list("loras"), {
                    "default": "none",
                    "tooltip": "Select LoRA file (optional)"
                }),
                "eligen_lora_file": (["none", "None", "NONE", "null", "Null", "NULL"] + folder_paths.get_filename_list("loras"), {
                    "default": "none",
                    "tooltip": "Select EliGen LoRA file (optional)"
                }),
            }
        }

    RETURN_TYPES = ("QWEN_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_pipeline(self, vram_optimization: str, offload_to_cpu: bool, torch_dtype: str,
                     lora: Optional[Dict] = None, controlnet: Optional[Dict] = None,
                     lightning_lora: str = "none",
                     lora_file: str = "none",
                     eligen_lora_file: str = "none",
                     base_model: str = "auto"):
        """Load Qwen-Image DiffSynth pipeline with optimizations."""
        try:
            # Convert torch_dtype string to torch dtype - proper FP8 support based on DiffSynth-Studio
            # Standard dtype handling
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if torch_dtype not in dtype_map:
                raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

            pipeline_dtype = dtype_map[torch_dtype]
            offload_dtype = None
            logger.info(f"Using dtype: {torch_dtype}")

            logger.info(f"Loading DiffSynth pipeline with {vram_optimization} optimization")
            if lora:
                logger.info(f"LoRA models: {lora.get('lora_name', [])}")
            if controlnet:
                logger.info(f"ControlNet: {controlnet.get('controlnet_name', 'Unknown')}")

            available, QwenImagePipeline, ModelConfig, ControlNetInput, ModelManager = _get_diffsynth_classes()

            if available:
                # Optional pre-load cleanup to reduce fragmentation (align with QwenImage-Diffsynth_8)
                try:
                    mm.unload_all_models()
                    mm.cleanup_models()
                    mm.cleanup_models_gc()
                except Exception:
                    pass

                # Build model_configs for DiffSynth-Studio with offload configuration
                model_configs = []
                off_dev = "cpu" if offload_to_cpu else "cuda"

                # Determine which base model to use (support Original / EliGen / Distill-Full)
                auto_load_eligen = False
                chosen = base_model or "auto"
                if chosen == "auto":
                    # 兼容旧逻辑：如选择了 lightning_lora，则优先蒸馏版以提速，否则走原版
                    if not _is_none_choice(lightning_lora):
                        chosen = "Qwen-Image-Distill-Full"
                    else:
                        chosen = "Qwen-Image"

                if chosen == "Qwen-Image-Distill-Full":
                    model_configs.append(
                        ModelConfig(
                            model_id="DiffSynth-Studio/Qwen-Image-Distill-Full",
                            origin_file_pattern="diffusion_pytorch_model*.safetensors",
                            offload_device=off_dev,
                            offload_dtype=offload_dtype
                        )
                    )
                    logger.info("Using base model: Qwen-Image-Distill-Full")
                elif chosen == "Qwen-Image":
                    model_configs.append(
                        ModelConfig(
                            model_id="Qwen/Qwen-Image",
                            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                            offload_device=off_dev,
                            offload_dtype=offload_dtype
                        )
                    )
                    logger.info("Using base model: Qwen-Image (original)")
                elif chosen == "Qwen-Image-EliGen":
                    # EliGen 作为实体控制的专用 LoRA；基座仍使用原版 Qwen-Image
                    model_configs.append(
                        ModelConfig(
                            model_id="Qwen/Qwen-Image",
                            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                            offload_device=off_dev,
                            offload_dtype=offload_dtype
                        )
                    )
                    auto_load_eligen = True
                    logger.info("Using base model: Qwen-Image + EliGen (LoRA)")
                else:
                    # 安全回退
                    model_configs.append(
                        ModelConfig(
                            model_id="Qwen/Qwen-Image",
                            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                            offload_device=off_dev,
                            offload_dtype=offload_dtype
                        )
                    )
                    logger.warning(f"Unknown base_model={base_model}, fallback to Qwen-Image")

                # Add text_encoder and VAE (always from original model)
                model_configs.extend([
                    ModelConfig(
                        model_id="Qwen/Qwen-Image",
                        origin_file_pattern="text_encoder/model*.safetensors",
                        offload_device=("cuda" if not controlnet else off_dev),
                        offload_dtype=offload_dtype
                    ),
                    ModelConfig(
                        model_id="Qwen/Qwen-Image",
                        origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                        offload_device=("cuda" if not controlnet else off_dev),
                        offload_dtype=offload_dtype
                    ),
                ])

                # 与参考实现一致：优先使用本地 ControlNet safetensors 路径注入到管线；
                # 若未提供路径，则按类型回退到官方仓库模型 ID。
                if controlnet:
                    controlnet_path = controlnet.get('controlnet_path')
                    controlnet_type = controlnet.get('controlnet_type')
                    if controlnet_path:
                        model_configs.append(
                            ModelConfig(
                                path=controlnet_path,
                                skip_download=True,
                                offload_device="cpu"
                            )
                        )
                        try:
                            basename = os.path.basename(controlnet_path)
                        except Exception:
                            basename = str(controlnet_path)
                        logger.info(f"Added ControlNet from local path: {basename} (offload_device=cpu)")
                    elif controlnet_type:
                        model_configs.append(
                            ModelConfig(
                                model_id=f"DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-{controlnet_type.title()}",
                                origin_file_pattern="model.safetensors",
                                offload_device="cpu"
                            )
                        )
                        logger.info(f"Added ControlNet config by type: {controlnet_type} (offload_device=cpu)")

                # Create tokenizer_config
                tokenizer_config = ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="tokenizer/",
                    offload_device=off_dev
                )

                # Ensure Inpaint ControlNet added by type when needed
                if controlnet and not controlnet.get('controlnet_path'):
                    cn_type = controlnet.get('controlnet_type')
                    if cn_type == 'inpaint':
                        try:
                            found_any = any(
                                getattr(mc, 'model_id', '').endswith('ControlNet-Inpaint')
                                for mc in model_configs
                            )
                        except Exception:
                            found_any = False
                        if not found_any:
                            model_configs.append(
                                ModelConfig(
                                    model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint",
                                    origin_file_pattern="model.safetensors",
                                    offload_device="cpu"
                                )
                            )
                            logger.info("Added official Inpaint ControlNet by type (fallback)")

                # Create DiffSynth pipeline using QwenImagePipeline.from_pretrained
                pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=pipeline_dtype,  # Use pipeline_dtype (bfloat16 for FP8, original for others)
                    device="cuda",
                    model_configs=model_configs,
                    tokenizer_config=tokenizer_config,
                )
                # 标记蒸馏模式（用于采样器正确限制步数）
                try:
                    if chosen == "Qwen-Image-Distill-Full":
                        setattr(pipe, "_augment_is_distill", True)
                except Exception:
                    pass

                # Apply VRAM optimization profile
                if vram_optimization == 'No_Optimization':
                    # No optimization: just enable basic VRAM management
                    pipe.enable_vram_management()
                    logger.info("Enabled pipeline built-in VRAM management")
                else:
                    try:
                        if MMGP_AVAILABLE:
                            # MMGP optimization
                            # Rule: No ControlNet -> keep TE/VAE on GPU and only offload transformer.
                            #       With ControlNet -> manage TE/VAE as original (off_dev), and include them in offload pipe when offloaded.
                            offload_pipe = (
                                {"transformer": pipe.dit} if (offload_to_cpu and not controlnet) else
                                {"transformer": pipe.dit, "text_encoder": pipe.text_encoder, "vae": pipe.vae}
                            )
                            if vram_optimization == 'HighRAM_HighVRAM':
                                opt = mmgp_profile_type.HighRAM_HighVRAM
                            elif vram_optimization == 'HighRAM_LowVRAM':
                                opt = mmgp_profile_type.HighRAM_LowVRAM
                            elif vram_optimization == 'LowRAM_HighVRAM':
                                opt = mmgp_profile_type.LowRAM_HighVRAM
                            elif vram_optimization == 'LowRAM_LowVRAM':
                                opt = mmgp_profile_type.LowRAM_LowVRAM
                            elif vram_optimization == 'VerylowRAM_LowVRAM':
                                opt = mmgp_profile_type.VerylowRAM_LowVRAM
                            else:
                                opt = mmgp_profile_type.HighRAM_LowVRAM
                            try:
                                mmgp_offload.profile(
                                    offload_pipe,
                                    opt,
                                    verboseLevel=1,
                                    pinnedMemory=True,
                                    asyncTransfers=True,
                                    budgets=(
                                        {"transformer": "90%"} if not controlnet else
                                        {"transformer": "65%", "text_encoder": "5%", "vae": "15%"}
                                    ),
                                    workingVRAM=("70%" if not controlnet else "25%"),
                                )
                            except TypeError:
                                # Older mmgp versions may not support the extra kwargs
                                mmgp_offload.profile(offload_pipe, opt, verboseLevel=1)
                            logger.info(f"Applied mmgp offload profile: {vram_optimization}")
                        else:
                            # MMGP not available: fallback to built-in management
                            pipe.enable_vram_management()
                            logger.info("MMGP not available, using pipeline built-in VRAM management")
                    except Exception as _e:
                        logger.warning(f"VRAM optimization failed: {_e}")
                        # Fallback to built-in management
                        pipe.enable_vram_management()
                        logger.info("Fallback: Enabled pipeline built-in VRAM management")

                # Load LoRA(s) selected via node inputs
                if lora and lora.get('lora_path'):
                    lora_paths = lora['lora_path'] if isinstance(lora['lora_path'], list) else [lora['lora_path']]
                    for lora_path in lora_paths:
                        pipe.load_lora(pipe.dit, lora_path)
                        # Detect Lightning LoRA even when provided via lora input node
                        try:
                            basename = os.path.basename(str(lora_path))
                            if "lightning" in basename.lower():
                                setattr(pipe, "_augment_has_lightning", True)
                                logger.info("Detected Lightning LoRA via lora input; flag set on pipeline")
                        except Exception:
                            pass
                        logger.info(f"Loaded LoRA: {lora_path}")

                # Direct dropdown-selected LoRA files
                try:
                    if not _is_none_choice(lightning_lora):
                        lp = folder_paths.get_full_path_or_raise("loras", lightning_lora)
                        pipe.load_lora(pipe.dit, lp)
                        try:
                            if "lightning" in os.path.basename(lp).lower():
                                setattr(pipe, "_augment_has_lightning", True)
                                logger.info("Detected Lightning LoRA; flag set on pipeline")
                        except Exception:
                            pass
                        logger.info(f"Loaded Lightning LoRA: {lp}")
                except Exception as e:
                    logger.warning(f"Failed to load Lightning LoRA {lightning_lora}: {e}")

                try:
                    if not _is_none_choice(lora_file):
                        lp = folder_paths.get_full_path_or_raise("loras", lora_file)
                        pipe.load_lora(pipe.dit, lp)
                        try:
                            if "lightning" in os.path.basename(lp).lower():
                                setattr(pipe, "_augment_has_lightning", True)
                                logger.info("Detected Lightning LoRA via lora_file; flag set on pipeline")
                        except Exception:
                            pass
                        logger.info(f"Loaded LoRA: {lp}")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA {lora_file}: {e}")

                try:
                    # 如果 base_model 选择 EliGen（auto_load_eligen=True），则自动尝试加载官方 EliGen LoRA（可选）
                    if 'auto_load_eligen' in locals() and auto_load_eligen:
                        try:
                            # 默认从本地 models/DiffSynth-Studio/Qwen-Image-EliGen/model.safetensors 加载
                            default_eligen_path = os.path.join("models", "DiffSynth-Studio", "Qwen-Image-EliGen", "model.safetensors")
                            if os.path.exists(default_eligen_path):
                                pipe.load_lora(pipe.dit, default_eligen_path)
                                logger.info(f"Auto-loaded EliGen LoRA: {default_eligen_path}")
                            else:
                                logger.warning("EliGen base selected but default EliGen LoRA path not found; please set eligen_lora_file")
                        except Exception as e:
                            logger.warning(f"Auto-load EliGen LoRA failed: {e}")
                    # 同时也支持用户手动选择 eligen_lora_file
                    if not _is_none_choice(eligen_lora_file):
                        lp = folder_paths.get_full_path_or_raise("loras", eligen_lora_file)
                        pipe.load_lora(pipe.dit, lp)
                        logger.info(f"Loaded EliGen LoRA: {lp}")
                except Exception as e:
                    logger.warning(f"Failed to load EliGen LoRA {eligen_lora_file}: {e}")

                # Load EliGen LoRA if needed (this is a special LoRA for entity control)
                # Note: EliGen is loaded separately as it's a specific LoRA model

                logger.info("DiffSynth pipeline loaded successfully")
                return (pipe,)


            else:
                # Make failure explicit instead of returning a mock pipeline
                msg = (
                    "DiffSynth-Studio is not available. Please install it first:\n"
                    "  git clone https://github.com/modelscope/DiffSynth-Studio.git\n"
                    "  cd DiffSynth-Studio\n"
                    "  pip install -e .\n"
                    "Then restart ComfyUI."
                )
                logger.error(msg)
                raise RuntimeError(msg)

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise RuntimeError(f"Pipeline loading failed: {e}")


class QwenImageDiffSynthControlNetInput:
    """
    Advanced ControlNet input processor for Qwen-Image DiffSynth pipeline.
    Supports multiple ControlNet types with intelligent parameter adjustment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image for ControlNet"
                }),
                "controlnet_type": (["canny", "depth", "pose", "normal", "seg", "inpaint"], {
                    "default": "canny",
                    "tooltip": "Type of ControlNet preprocessing"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "ControlNet influence strength"
                }),
                "preprocessing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable automatic preprocessing"
                })
            },
            "optional": {
                # Canny specific parameters
                "canny_low_threshold": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Canny low threshold (only for canny type)"
                }),
                "canny_high_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Canny high threshold (only for canny type)"
                }),
                # Depth specific parameters
                "depth_near_plane": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Depth near plane (only for depth type)"
                }),
                "depth_far_plane": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "tooltip": "Depth far plane (only for depth type)"
                }),
                # Pose specific parameters
                "pose_detect_hands": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Detect hands in pose (only for pose type)"
                }),
                "pose_detect_face": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Detect face in pose (only for pose type)"
                }),
                # General parameters
                "resize_mode": (["resize", "crop", "pad"], {
                    "default": "resize",
                    "tooltip": "How to handle image resizing"
                }),
                "invert_image": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the processed image"
                })
            }
        }

    RETURN_TYPES = ("CONTROL_ARGS", "IMAGE")
    RETURN_NAMES = ("control_args", "preview_image")
    FUNCTION = "prepare_controlnet_input"
    CATEGORY = "Ken-Chen/Qwen-Image/conditioning"

    def prepare_controlnet_input(self, image, controlnet_type: str, strength: float,
                               preprocessing: bool,
                               canny_low_threshold: int = 100,
                               canny_high_threshold: int = 200,
                               depth_near_plane: float = 0.1,
                               depth_far_plane: float = 100.0,
                               pose_detect_hands: bool = True,
                               pose_detect_face: bool = True,
                               resize_mode: str = "resize",
                               invert_image: bool = False):
        """
        Advanced ControlNet input preparation with type-specific parameters.
        Automatically adjusts parameters based on ControlNet type.
        """
        try:
            processed_image = image
            preview_image = image

            # Build parameters based on ControlNet type
            params = self._build_params_for_type(
                controlnet_type,
                canny_low_threshold, canny_high_threshold,
                depth_near_plane, depth_far_plane,
                pose_detect_hands, pose_detect_face,
                resize_mode, invert_image
            )

            if preprocessing:
                # Apply type-specific preprocessing
                processed_image, preview_image = self._apply_preprocessing(
                    image, controlnet_type, params
                )

            # Create ControlNet input configuration
            controlnet_input = {
                "image": processed_image,
                "type": controlnet_type,
                "strength": strength,
                "params": params,
                "preprocessing_applied": preprocessing
            }

            logger.info(f"Prepared ControlNet input: type={controlnet_type}, strength={strength}, preprocessing={preprocessing}")
            return (controlnet_input, preview_image)

        except Exception as e:
            logger.error(f"Failed to prepare ControlNet input: {e}")
            raise RuntimeError(f"ControlNet input preparation failed: {e}")

    def _build_params_for_type(self, controlnet_type: str,
                              canny_low: int, canny_high: int,
                              depth_near: float, depth_far: float,
                              pose_hands: bool, pose_face: bool,
                              resize_mode: str, invert: bool):
        """Build parameters specific to ControlNet type."""
        base_params = {
            "resize_mode": resize_mode,
            "invert_image": invert
        }

        if controlnet_type == "canny":
            base_params.update({
                "low_threshold": canny_low,
                "high_threshold": canny_high
            })
        elif controlnet_type == "depth":
            base_params.update({
                "near_plane": depth_near,
                "far_plane": depth_far
            })
        elif controlnet_type == "pose":
            base_params.update({
                "detect_hands": pose_hands,
                "detect_face": pose_face
            })
        elif controlnet_type == "normal":
            base_params.update({
                "background_threshold": 0.4
            })
        elif controlnet_type == "seg":
            base_params.update({
                "num_classes": 150
            })

        return base_params

    def _apply_preprocessing(self, image, controlnet_type: str, params: dict):
        """
        Apply type-specific preprocessing to the image.
        Returns both processed image and preview image.
        """
        import cv2
        import numpy as np

        # Convert ComfyUI tensor to numpy array
        if len(image.shape) == 4:  # Batch dimension
            img_np = image[0].cpu().numpy()
        else:
            img_np = image.cpu().numpy()

        # Convert to uint8 format for OpenCV
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # Ensure RGB format
        if img_np.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np

        processed_img = None

        try:
            if controlnet_type == "canny":
                processed_img = self._apply_canny(img_bgr, params)
            elif controlnet_type == "depth":
                processed_img = self._apply_depth(img_bgr, params)
            elif controlnet_type == "pose":
                processed_img = self._apply_pose(img_bgr, params)
            elif controlnet_type == "normal":
                processed_img = self._apply_normal(img_bgr, params)
            elif controlnet_type == "seg":
                processed_img = self._apply_segmentation(img_bgr, params)
            else:
                # For other types, use original image
                processed_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e:
            logger.error(f"Error in {controlnet_type} preprocessing: {e}")
            # Fallback to original image
            processed_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert back to ComfyUI tensor format
        processed_tensor = torch.from_numpy(processed_img.astype(np.float32) / 255.0)
        if len(processed_tensor.shape) == 3:
            processed_tensor = processed_tensor.unsqueeze(0)  # Add batch dimension

        logger.info(f"Applied {controlnet_type} preprocessing with params: {params}")

        return processed_tensor, processed_tensor

    def _apply_canny(self, img_bgr, params):
        """Apply Canny edge detection with improved parameters."""
        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Get thresholds from parameters
        low_threshold = params.get("low_threshold", 100)
        high_threshold = params.get("high_threshold", 200)

        # Ensure high threshold is greater than low threshold
        if high_threshold <= low_threshold:
            high_threshold = low_threshold + 100

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # Convert to RGB format
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        logger.info(f"Canny edge detection applied: low={low_threshold}, high={high_threshold}")
        return edges_rgb

    def _apply_depth(self, img_bgr, params):
        """Apply depth estimation using MiDaS."""
        import cv2
        try:
            # Try to use controlnet_aux for better depth estimation
            from controlnet_aux import MidasDetector
            midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            depth_map = midas(img_rgb)

            # Ensure depth_map is in the correct format
            depth_array = np.array(depth_map)

            # If it's grayscale, convert to RGB
            if len(depth_array.shape) == 2:
                depth_rgb = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2RGB)
            elif len(depth_array.shape) == 3 and depth_array.shape[2] == 1:
                depth_rgb = cv2.cvtColor(depth_array.squeeze(), cv2.COLOR_GRAY2RGB)
            else:
                depth_rgb = depth_array

            # Ensure it's the same size as input
            if depth_rgb.shape[:2] != img_bgr.shape[:2]:
                depth_rgb = cv2.resize(depth_rgb, (img_bgr.shape[1], img_bgr.shape[0]))

            return depth_rgb

        except ImportError:
            logger.warning("controlnet_aux not available, using simple depth approximation")
            # Fallback: use edge-based depth approximation
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to simulate depth
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            # Invert so closer objects are brighter
            depth_approx = 255 - blurred
            return cv2.cvtColor(depth_approx, cv2.COLOR_GRAY2RGB)

    def _apply_pose(self, img_bgr, params):
        """Apply pose detection using MediaPipe or OpenPose."""
        import cv2
        try:
            # Try MediaPipe first
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils

            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            ) as pose:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                # Create black canvas
                pose_img = np.zeros_like(img_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        pose_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                return pose_img

        except ImportError:
            logger.warning("MediaPipe not available, trying controlnet_aux")
            try:
                from controlnet_aux import OpenposeDetector
                openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pose_img = openpose(img_rgb,
                                  hand=params.get("detect_hands", True),
                                  face=params.get("detect_face", True))
                return np.array(pose_img)
            except ImportError:
                logger.warning("No pose detection library available, using edge detection")
                # Fallback: use edge detection as pose approximation
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def _apply_normal(self, img_bgr, params):
        """Apply normal map generation."""
        import cv2
        try:
            from controlnet_aux import NormalBaeDetector
            normal_detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            normal_map = normal_detector(img_rgb)
            return np.array(normal_map)
        except ImportError:
            logger.warning("controlnet_aux not available, using gradient-based normal approximation")
            # Fallback: create normal map from gradients
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Normalize gradients
            grad_x = grad_x / 255.0
            grad_y = grad_y / 255.0

            # Create normal map (simplified)
            normal_x = (grad_x + 1) * 127.5
            normal_y = (grad_y + 1) * 127.5
            normal_z = np.full_like(normal_x, 255)  # Z component

            normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
            normal_map = np.clip(normal_map, 0, 255).astype(np.uint8)

            return normal_map

    def _apply_segmentation(self, img_bgr, params):
        """Apply semantic segmentation."""
        import cv2
        try:
            from controlnet_aux import SamDetector
            sam_detector = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            seg_map = sam_detector(img_rgb)
            return np.array(seg_map)
        except ImportError:
            logger.warning("SAM not available, trying simple segmentation")
            try:
                # Fallback: use watershed segmentation
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # Apply threshold to get binary image
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Remove noise
                kernel = np.ones((3,3), np.uint8)
                opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

                # Sure background area
                sure_bg = cv2.dilate(opening, kernel, iterations=3)

                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

                # Create segmentation map with different colors
                seg_map = np.zeros_like(img_bgr)
                seg_map[sure_bg == 255] = [128, 128, 128]  # Background
                seg_map[sure_fg == 255] = [255, 255, 255]  # Foreground

                return cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)

            except Exception as e:
                logger.warning(f"Segmentation failed: {e}, using edge detection")
                # Final fallback
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


class QwenDiffSynthSetControlArgs:
    """
    Convert preprocessed ControlNet images to control_args format.
    Similar to SetControlArgs in QwenImage-Diffsynth_8.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
            },
            "optional": {
                "inpaint_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONTROL_ARGS",)
    RETURN_NAMES = ("control_args",)
    FUNCTION = "set_control_args"
    CATEGORY = "Ken-Chen/Qwen-Image/conditioning"

    def set_control_args(self, ref_image, inpaint_mask=None):
        """
        Convert preprocessed ControlNet image to control_args format.
        This allows using existing ControlNet preprocessing nodes.
        """
        try:
            # Use the same import pattern as other nodes in this file
            _, _, _, ControlNetInput, _ = _get_diffsynth_classes()

            # Convert ComfyUI tensor to PIL Image
            def comfy2pil(tensor):
                import numpy as np
                from PIL import Image

                if len(tensor.shape) == 4:
                    tensor = tensor[0]  # Remove batch dimension

                # Convert to numpy
                img_np = tensor.cpu().numpy()

                # Convert to uint8
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

                return Image.fromarray(img_np)

            # Convert ref_image to PIL
            ref_pil = comfy2pil(ref_image)

            # Convert inpaint_mask if provided
            inpaint_pil = None
            if inpaint_mask is not None:
                inpaint_pil = comfy2pil(inpaint_mask)

            # Create control_args in the format expected by Advanced Sampler
            # Advanced Sampler expects a list of dicts with 'image' key
            control_dict = {
                'image': ref_pil,
            }

            if inpaint_pil is not None:
                control_dict['inpaint_mask'] = inpaint_pil

            # Return as list (matching the expected format)
            control_args = [control_dict]

            logger.info(f"Created control_args from preprocessed image: {ref_pil.size}")
            return (control_args,)

        except Exception as e:
            logger.error(f"Error creating control_args: {e}")
            raise e


class QwenImageDiffSynthAdvancedSampler:
    """
    Advanced DiffSynth sampler for Qwen-Image with enhanced features.
    Supports EliGen entity control and blockwise ControlNet integration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_PIPELINE", {
                    "tooltip": "Qwen-Image DiffSynth pipeline"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的小猫咪坐在樱花树下，春天的阳光洒在它的毛发上，背景是传统的中式庭院，超清，4K，电影级构图，细节精致，梦幻唯美",
                    "tooltip": "Text prompt for generation"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, low quality, distorted, ugly, bad anatomy",
                    "tooltip": "Negative prompt"
                }),
                "width": ("INT", {
                    "default": 1328,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Image width"
                }),
                "height": ("INT", {
                    "default": 1328,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Image height"
                }),
                "num_inference_steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps (higher = better quality)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 8.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG guidance scale (higher = more prompt adherence)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed"
                }),
                "sampling_method": (["default", "enhanced_quality", "fast"], {
                    "default": "enhanced_quality",
                    "tooltip": "Sampling method: default (standard), enhanced_quality (slower but better), fast (quicker)"
                }),
                "clip_skip": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 3,
                    "tooltip": "Skip last N layers of CLIP (0 = use all layers)"
                }),
                "enhance_quality": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply post-processing to enhance image quality"
                }),
                "sharpen_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Sharpening strength (0.0 = no sharpening)"
                }),
                "color_enhance": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Color saturation enhancement (1.0 = no change)"
                }),

            },
            "optional": {
                "eligen_args": ("ELIGEN_ARGS", {
                    "tooltip": "EliGen entity arguments (optional)"
                }),
                "control_args": ("CONTROL_ARGS", {
                    "tooltip": "ControlNet arguments (optional)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "Ken-Chen/Qwen-Image/sampling"

    def sample(self, pipeline, prompt: str, negative_prompt: str,
               width: int, height: int, num_inference_steps: int,
               guidance_scale: float, seed: int, sampling_method: str = "enhanced_quality",
               clip_skip: int = 0, enhance_quality: bool = True,
               sharpen_strength: float = 0.3, color_enhance: float = 1.1,
               eligen_args: Optional[Dict] = None,
               control_args: Optional[List] = None):
        """
        Generate image using DiffSynth pipeline with proper implementation.
        """
        try:
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            logger.info(f"DiffSynth Advanced Sampler generating: {prompt[:50]}...")
            logger.info(f"Size: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")

            # Check if we have a real DiffSynth pipeline
            if hasattr(pipeline, '__call__'):
                # Real DiffSynth pipeline call following official API
                generation_kwargs = {
                    "prompt": prompt,
                    "seed": seed,
                }
                # Ensure ControlNetInput class is available in this scope
                # (comes from DiffSynth or our safe fallback)
                _, _, _, ControlNetInput, _ = _get_diffsynth_classes()

                # Handle different inference modes based on model type and Lightning LoRA
                has_lightning = bool(getattr(pipeline, "_augment_has_lightning", False))
                is_distill = bool(getattr(pipeline, "_augment_is_distill", False))
                if not is_distill and hasattr(pipeline, 'dit'):
                    # 兼容兜底：有些环境类名不含 Distill，这里用名称兜底一次
                    try:
                        is_distill = 'Distill' in type(pipeline.dit).__name__
                    except Exception:
                        is_distill = False

                if has_lightning:
                    # Lightning 模式：尊重用户的步数设置（支持 4/8 步等），并允许使用 enhanced_quality
                    base_kwargs = {
                        "cfg_scale": guidance_scale,
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": num_inference_steps,
                    }
                    if sampling_method == "enhanced_quality":
                        # 仅调整 CFG，不强制提升步数
                        base_kwargs.update({
                            "cfg_scale": min(guidance_scale * 1.1, 15.0)
                        })
                        logger.info("Lightning + enhanced_quality: applying CFG boost only; keep user steps")
                    elif sampling_method == "fast":
                        base_kwargs.update({
                            "cfg_scale": max(guidance_scale * 0.9, 4.0)
                        })
                        logger.info("Lightning + fast: slightly lower CFG; keep user steps")
                    else:
                        logger.info("Lightning: using user steps and CFG as-is")

                    generation_kwargs.update(base_kwargs)
                elif is_distill:
                    # Distilled model - optimized settings; NEVER increase steps due to sampling_method
                    generation_kwargs.update({
                        "cfg_scale": 1.0,  # Distilled models don't need high CFG
                        "num_inference_steps": min(num_inference_steps, 15),  # Respect user steps but cap at 15
                    })
                    logger.info("Using distilled model optimized settings (cfg_scale=1, steps<=15; keep user steps)")
                else:
                    # Original model - full settings with quality enhancements
                    base_kwargs = {
                        "cfg_scale": guidance_scale,
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": num_inference_steps,
                    }

                    # Apply sampling method optimizations
                    if sampling_method == "enhanced_quality":
                        # Higher quality settings
                        base_kwargs.update({
                            "cfg_scale": min(guidance_scale * 1.1, 15.0),  # Slightly higher CFG
                            "num_inference_steps": max(num_inference_steps, 25),  # Minimum 25 steps
                        })
                        logger.info("Using enhanced quality sampling (higher CFG, more steps)")
                    elif sampling_method == "fast":
                        # Faster settings
                        base_kwargs.update({
                            "cfg_scale": max(guidance_scale * 0.9, 4.0),  # Slightly lower CFG
                            "num_inference_steps": min(num_inference_steps, 20),  # Maximum 20 steps
                        })
                        logger.info("Using fast sampling (lower CFG, fewer steps)")
                    else:
                        logger.info("Using default sampling settings")

                    generation_kwargs.update(base_kwargs)

                # Always pass explicit image dimensions and align to DiT-friendly grid
                # Qwen-Image DiT uses 16x16 patches; some modules (e.g., Blockwise ControlNet)
                # expect the token grid (W/16 x H/16) to be multiples of 32.
                def _align_dit_grid(x: int, patch: int = 16, grid_multiple: int = 32) -> int:
                    # Round the token grid to nearest multiple of `grid_multiple`, then map back to pixels
                    grid = max(1, x // patch)
                    lower = (grid // grid_multiple) * grid_multiple
                    upper = lower + grid_multiple
                    # choose nearer multiple; prefer lower on tie to save VRAM
                    chosen = lower if (grid - lower) <= (upper - grid) else upper
                    chosen = max(grid_multiple, chosen)
                    return chosen * patch

                safe_width = _align_dit_grid(width)
                safe_height = _align_dit_grid(height)
                if (safe_width, safe_height) != (width, height):
                    logger.warning(f"Adjusting size to DiT grid (tokens multiple of 32): {width}x{height} -> {safe_width}x{safe_height}")

                generation_kwargs.update({
                    "height": safe_height,
                    "width": safe_width,
                })

                # Add EliGen entity control if provided (following official EliGen example)
                if eligen_args and eligen_args.get("masks") is not None:
                    masks = eligen_args.get("masks")
                    entity_prompts = eligen_args.get("prompts")
                    global_prompt = eligen_args.get("global_prompt")

                    # Sanitize entity text prompts like "字：……" before passing to pipeline
                    if isinstance(entity_prompts, list):
                        entity_prompts = [_sanitize_text_prompt(p) if isinstance(p, str) else p for p in entity_prompts]


                    # If global_prompt exists, merge it into main prompt
                    if isinstance(global_prompt, str) and global_prompt.strip():
                        try:
                            base_prompt = generation_kwargs.get("prompt", prompt)
                            gp = _sanitize_text_prompt(global_prompt)
                            # 若用户确实意图生成文字（例如只输入了“炮老师咖啡|新品上市”），则直接合并，不要带“字:”
                            merged = f"{gp}, {base_prompt}" if base_prompt else gp
                            generation_kwargs["prompt"] = merged
                            logger.info("Merged EliGen global_prompt (sanitized) into prompt")
                        except Exception as _e:
                            logger.warning(f"Failed to merge EliGen global_prompt: {_e}")

                    if masks is not None and entity_prompts is not None:
                        # Convert ComfyUI tensors to PIL Images if needed
                        if isinstance(masks, list):
                            pil_masks = []
                            for mask in masks:
                                # Convert Tensor to PIL
                                if isinstance(mask, torch.Tensor):
                                    mask = comfy2pil(mask)
                                # Ensure PIL.Image
                                if not isinstance(mask, Image.Image):
                                    try:
                                        mask = Image.fromarray(np.array(mask))
                                    except Exception:
                                        pass
                                # Ensure 3-channel RGB for masks to satisfy H W C expectations
                                if isinstance(mask, Image.Image) and mask.mode != 'RGB':
                                    try:
                                        mask = mask.convert('RGB')
                                    except Exception:
                                        pass
                                # Resize to generation resolution to ensure alignment
                                try:
                                    if isinstance(mask, Image.Image) and mask.size != (safe_width, safe_height):
                                        mask = mask.resize((safe_width, safe_height), Image.NEAREST)
                                except Exception:
                                    pass
                                pil_masks.append(mask)
                            masks = pil_masks

                        generation_kwargs.update({
                            "eligen_entity_masks": masks,
                            "eligen_entity_prompts": entity_prompts,
                        })
                        try:
                            _m0 = masks[0] if isinstance(masks, list) and masks else None
                            _ms = getattr(_m0, 'size', None)
                            _mm = getattr(_m0, 'mode', None)
                            logger.info(f"Using EliGen entity control with {len(entity_prompts)} entities; first mask size={_ms} mode={_mm}")
                        except Exception:
                            logger.info(f"Using EliGen entity control with {len(entity_prompts)} entities")

                # Add ControlNet inputs if provided (following official ControlNet example)
                if control_args:
                    # Accept both dict (single) and list/tuple of dicts
                    if isinstance(control_args, dict):
                        control_args_iter = [control_args]
                    elif isinstance(control_args, (list, tuple)):
                        control_args_iter = control_args
                    else:
                        control_args_iter = []

                    blockwise_controlnet_inputs = []

                    for control_arg in control_args_iter:
                        if isinstance(control_arg, dict) and 'image' in control_arg:
                            # Convert ComfyUI tensor to PIL if needed
                            control_image = control_arg['image']
                            if isinstance(control_image, torch.Tensor):
                                control_image = comfy2pil(control_image)

                            # Ensure control image matches generation resolution and format
                            try:
                                if hasattr(control_image, 'mode') and control_image.mode != 'RGB':
                                    control_image = control_image.convert('RGB')
                                if control_image.size != (safe_width, safe_height):
                                    old_w, old_h = control_image.size
                                    control_image = control_image.resize((safe_width, safe_height), Image.BICUBIC)
                                    logger.info(f"Resized ControlNet image from {old_w}x{old_h} to {safe_width}x{safe_height} to match DiT grid")
                            except Exception as _e:
                                logger.warning(f"Failed to normalize ControlNet image size/mode: {_e}")

                            # Create ControlNetInput following official API
                            # If inpaint mask is provided in control args, pass it through
                            if isinstance(control_arg, dict) and 'inpaint_mask' in control_arg and control_arg['inpaint_mask'] is not None:
                                mask_img = control_arg['inpaint_mask']
                                # Normalize size/mode for mask as well
                                try:
                                    if isinstance(mask_img, torch.Tensor):
                                        mask_img = mask_img.squeeze(0).clamp(0, 1)
                                        mask_img = (mask_img.cpu().numpy() * 255).astype('uint8')
                                        if mask_img.ndim == 3 and mask_img.shape[2] == 3:
                                            mask_img = Image.fromarray(mask_img)
                                        else:
                                            mask_img = Image.fromarray(mask_img).convert("L")
                                    elif isinstance(mask_img, np.ndarray):
                                        mask_img = Image.fromarray(mask_img)
                                    elif not isinstance(mask_img, Image.Image):
                                        # try attribute .image
                                        if hasattr(mask_img, 'image'):
                                            mask_img = mask_img.image
                                        if not isinstance(mask_img, Image.Image):
                                            raise TypeError("Unsupported inpaint_mask type")
                                    # Ensure mask size matches control image
                                    if mask_img.size != control_image.size:
                                        mask_img = mask_img.resize(control_image.size, Image.BILINEAR)
                                except Exception as _e:
                                    logger.warning(f"Failed to normalize inpaint mask: {_e}")
                                    mask_img = None
                                controlnet_input = ControlNetInput(image=control_image, inpaint_mask=mask_img)
                            else:
                                controlnet_input = ControlNetInput(image=control_image)
                            blockwise_controlnet_inputs.append(controlnet_input)

                    if blockwise_controlnet_inputs:
                        # Safety: only pass ControlNet inputs if pipeline actually loaded ControlNet weights
                        if getattr(pipeline, 'blockwise_controlnet', None) is None:
                            logger.warning("ControlNet inputs provided but pipeline has no Blockwise ControlNet loaded; ignoring control_args to avoid crash")
                        else:
                            generation_kwargs["blockwise_controlnet_inputs"] = blockwise_controlnet_inputs
                            logger.info(f"Using ControlNet with {len(blockwise_controlnet_inputs)} control image(s)")

                # Generate image using DiffSynth pipeline (official API call)
                # IMPORTANT: Do NOT move modules to CUDA here; let DiffSynth's own
                # enable_vram_management/mmgp profile control device placement to keep VRAM low.

                logger.info("Calling DiffSynth pipeline for generation...")
                logger.info(f"Generation parameters: {list(generation_kwargs.keys())}")

                # Try to release any lingering cache before the heavy call
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass

                try:
                    image = pipeline(**generation_kwargs)
                except TypeError as te:
                    msg = str(te)
                    if "unexpected keyword argument 'height'" in msg or "unexpected keyword argument 'width'" in msg:
                        logger.warning("Pipeline does not accept width/height; retrying without explicit size")
                        generation_kwargs.pop("height", None)
                        generation_kwargs.pop("width", None)
                        image = pipeline(**generation_kwargs)
                    else:
                        raise
                except RuntimeError as re:
                    msg = str(re)
                    if "Expected all tensors to be on the same device" in msg:
                        logger.warning("Device mismatch detected. Attempting to align model devices and retry...")
                        # When using mmgp offload, models are on CPU and need to be moved to CUDA for inference
                        # This is a common issue when mmgp profile is applied but inference tensors are on different devices
                        try:
                            # Force all pipeline components to CUDA for inference (enhanced for Distill models)
                            if hasattr(pipeline, 'dit') and pipeline.dit is not None:
                                pipeline.dit.to('cuda')
                                # Also move any buffers/parameters that might be on CPU
                                for param in pipeline.dit.parameters():
                                    if param.device.type == 'cpu':
                                        param.data = param.data.to('cuda')
                                for buffer in pipeline.dit.buffers():
                                    if buffer.device.type == 'cpu':
                                        buffer.data = buffer.data.to('cuda')
                            if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                                pipeline.text_encoder.to('cuda')
                            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                                pipeline.vae.to('cuda')
                            if hasattr(pipeline, 'blockwise_controlnet') and pipeline.blockwise_controlnet is not None:
                                pipeline.blockwise_controlnet.to('cuda')
                            logger.info("Moved all pipeline components to CUDA for inference")
                        except Exception as _e:
                            logger.debug(f"Device realignment attempt encountered an issue: {_e}")
                        # Retry once
                        image = pipeline(**generation_kwargs)
                    else:
                        raise

                # Apply quality enhancement if requested
                if enhance_quality and (sharpen_strength > 0.0 or color_enhance != 1.0):
                    if hasattr(image, 'convert'):  # PIL Image
                        image = self._enhance_image_quality(image, sharpen_strength, color_enhance)
                        logger.info(f"Applied quality enhancement: sharpen={sharpen_strength}, color={color_enhance}")

                # Convert to ComfyUI tensor format
                if hasattr(image, 'convert'):  # PIL Image
                    image_tensor = pil2comfy(image)
                elif isinstance(image, torch.Tensor):
                    # Already a tensor, ensure correct format
                    if image.dim() == 3:
                        image_tensor = image.unsqueeze(0)  # Add batch dimension
                    else:
                        image_tensor = image
                else:
                    raise RuntimeError(f"Unexpected image type: {type(image)}")

                logger.info("DiffSynth Advanced Sampler: Image generated successfully")
                return (image_tensor,)

            else:
                # Produce a clear error image instead of a mock gradient
                err = "DiffSynth pipeline is not available. Please install DiffSynth-Studio and reload."
                logger.error(err)
                error_image = self._create_error_image(width, height, err)
                return (error_image,)

        except Exception as e:
            logger.error(f"DiffSynth Advanced Sampler failed: {e}")
            # Create a meaningful error image instead of crashing
            error_image = self._create_error_image(width, height, str(e))
            return (error_image,)

    def _enhance_image_quality(self, pil_image, sharpen_strength: float, color_enhance: float):
        """
        Apply post-processing to enhance image quality.
        """
        try:
            from PIL import Image, ImageEnhance, ImageFilter

            enhanced_image = pil_image.copy()

            # Apply sharpening
            if sharpen_strength > 0.0:
                # Create unsharp mask effect
                blurred = enhanced_image.filter(ImageFilter.GaussianBlur(radius=1.0))
                # Blend original with inverted blur for sharpening
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(1.0 + sharpen_strength)
                logger.debug(f"Applied sharpening with strength {sharpen_strength}")

            # Apply color enhancement
            if color_enhance != 1.0:
                enhancer = ImageEnhance.Color(enhanced_image)
                enhanced_image = enhancer.enhance(color_enhance)
                logger.debug(f"Applied color enhancement with factor {color_enhance}")

            # Apply subtle contrast enhancement
            if sharpen_strength > 0.0 or color_enhance > 1.0:
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(1.05)  # Subtle contrast boost
                logger.debug("Applied subtle contrast enhancement")

            return enhanced_image

        except Exception as e:
            logger.warning(f"Quality enhancement failed: {e}, returning original image")
            return pil_image

    def _create_mock_image(self, width: int, height: int, prompt: str, seed: int):
        """Generate a mock image for testing purposes."""
        # Create gradient background based on prompt and seed
        image_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Use prompt hash and seed for deterministic colors
        prompt_hash = hash(prompt) % 256
        seed_val = seed % 256

        # Create a colorful gradient
        for y in range(height):
            for x in range(width):
                image_array[y, x, 0] = int((x / width) * 255) ^ prompt_hash  # Red gradient
                image_array[y, x, 1] = int((y / height) * 255) ^ seed_val    # Green gradient
                image_array[y, x, 2] = int(((x + y) / (width + height)) * 255) ^ (prompt_hash + seed_val) % 256  # Blue gradient

        # Convert to ComfyUI tensor format
        image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        return image_tensor

    def _create_error_image(self, width: int, height: int, error_msg: str):
        """Create an error image with diagnostic information."""
        import cv2

        # Create a red-tinted background to indicate error
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Red gradient background
        for y in range(height):
            ratio = y / height
            image[y, :, 0] = int(150 + 105 * ratio)  # Red
            image[y, :, 1] = int(50 + 25 * ratio)    # Green
            image[y, :, 2] = int(50 + 25 * ratio)    # Blue

        # Add error message
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 800.0  # Scale font with image size

        # Main error title
        title = "DIFFSYNTH GENERATION ERROR"
        title_size = cv2.getTextSize(title, font, font_scale, 2)[0]
        title_x = (width - title_size[0]) // 2
        title_y = height // 3
        cv2.putText(image, title, (title_x, title_y), font, font_scale, (255, 255, 255), 2)

        # Error details (truncated if too long)
        error_text = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
        error_lines = error_text.split('\n')[:3]  # Max 3 lines

        for i, line in enumerate(error_lines):
            line_size = cv2.getTextSize(line, font, font_scale * 0.6, 1)[0]
            line_x = (width - line_size[0]) // 2
            line_y = title_y + 60 + (i * 30)
            cv2.putText(image, line, (line_x, line_y), font, font_scale * 0.6, (255, 255, 255), 1)

        # Add troubleshooting hint
        hint = "Check pipeline and dependencies"
        hint_size = cv2.getTextSize(hint, font, font_scale * 0.5, 1)[0]
        hint_x = (width - hint_size[0]) // 2
        hint_y = height - 50
        cv2.putText(image, hint, (hint_x, hint_y), font, font_scale * 0.5, (200, 200, 200), 1)

        # Convert to ComfyUI tensor format
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        return image_tensor


class QwenImageDiffSynthMemoryManager:
    """
    Advanced memory management for Qwen-Image DiffSynth.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["clear_cache", "optimize_memory", "get_memory_info"], {
                    "default": "optimize_memory",
                    "tooltip": "Memory management action"
                }),
                "aggressive_cleanup": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Perform aggressive memory cleanup"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("memory_info",)
    FUNCTION = "manage_memory"
    CATEGORY = "Ken-Chen/Qwen-Image/utils"

    def manage_memory(self, action: str, aggressive_cleanup: bool):
        """Manage memory for optimal performance."""
        try:
            import gc

            if action == "clear_cache":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                info = "CUDA cache cleared and garbage collected"

            elif action == "optimize_memory":
                # Optimize memory usage
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if aggressive_cleanup:
                        torch.cuda.synchronize()
                gc.collect()
                info = f"Memory optimized (aggressive: {aggressive_cleanup})"

            elif action == "get_memory_info":
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    info = f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                else:
                    info = "CUDA not available"

            logger.info(f"Memory management: {info}")
            return (info,)

        except Exception as e:
            logger.error(f"Memory management failed: {e}")
            return (f"Error: {e}",)


class QwenVRAMCleanupPassThrough:
    """
    可串接的显存清理直通节点。
    - 接收任意类型输入（anything），执行显存/模型卸载/缓存清理后，原样返回输入，便于插入任意链路。
    - offload_model: 卸载已加载模型并进行 ComfyUI 级清理（可能导致后续节点重新加载模型，慎用）。
    - offload_cache: 清空 CUDA cache + Python 垃圾回收，风险较小，推荐常用。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (ANY, {}),
                "offload_model": ("BOOLEAN", {"default": True}),
                "offload_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "cleanup_and_pass"
    CATEGORY = "Ken-Chen/Qwen-Image/utils"

    def cleanup_and_pass(self, anything, offload_model: bool = True, offload_cache: bool = True):
        try:
            import gc
            # 卸载模型并进行 ComfyUI 模型管理层的清理
            if offload_model:
                try:
                    mm.unload_all_models()
                    mm.cleanup_models()
                    mm.cleanup_models_gc()
                    logger.info("VRAM-Cleanup: unloaded models and cleaned model manager caches")
                except Exception as e:
                    logger.warning(f"VRAM-Cleanup: model offload failed: {e}")
            # 清空 CUDA cache 与 Python GC
            if offload_cache:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("VRAM-Cleanup: cleared CUDA cache and ran GC")
                except Exception as e:
                    logger.warning(f"VRAM-Cleanup: cache clear failed: {e}")
        except Exception as e:
            logger.warning(f"VRAM-Cleanup encountered an error: {e}")
        # 直通返回输入，保证链路不断裂
        return (anything,)


class QwenMemoryManagerPassThrough:
    """
    可串接的内存管理直通节点（保持 Qwen DiffSynth Memory Manager 的功能与选项）。
    - 输入任意类型 anything，执行 memory action 后原样返回 anything，另输出 memory_info 便于日志/显示。
    - action: clear_cache / optimize_memory / get_memory_info
    - aggressive_cleanup: 仅在 optimize_memory 时更彻底地清理（torch.cuda.synchronize + GC）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (ANY, {}),
                "action": (["clear_cache", "optimize_memory", "get_memory_info"], {"default": "optimize_memory"}),
                "aggressive_cleanup": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("output", "memory_info")
    FUNCTION = "manage_and_pass"
    CATEGORY = "Ken-Chen/Qwen-Image/utils"

    def manage_and_pass(self, anything, action: str = "optimize_memory", aggressive_cleanup: bool = False):
        try:
            import gc
            info = ""
            if action == "clear_cache":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                info = "CUDA cache cleared and garbage collected"
            elif action == "optimize_memory":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if aggressive_cleanup:
                        torch.cuda.synchronize()
                gc.collect()
                info = f"Memory optimized (aggressive: {aggressive_cleanup})"
            elif action == "get_memory_info":
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    info = f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                else:
                    info = "CUDA not available"
            else:
                info = f"Unknown action: {action}"
            logger.info(f"Memory management (pass-through): {info}")
            return (anything, info)
        except Exception as e:
            logger.error(f"Memory management (pass-through) failed: {e}")
            return (anything, f"Error: {e}")


# Utility functions inspired by QwenImage-Diffsynth_8
def comfy2pil(image):
    """Convert ComfyUI image tensor to PIL Image."""
    i = 255. * image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

def pil2comfy(pil):
    """Convert PIL Image to ComfyUI image tensor."""
    image = np.array(pil).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image


class QwenImageDiffSynthRatio2Size:
    """
    DiffSynth aspect ratio to size converter for Qwen-Image.
    Based on standard DiffSynth dimensions for optimal generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "1:1",
                    "tooltip": "Select aspect ratio"
                })
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_image_size"
    CATEGORY = "Ken-Chen/Qwen-Image/utils"

    def get_image_size(self, aspect_ratio: str):
        """Get image dimensions based on aspect ratio."""
        # Prefer DiT-friendly dimensions (both sides divisible by 512; token grid multiples of 32)
        if aspect_ratio == "1:1":
            return (1024, 1024)
        elif aspect_ratio == "16:9":
            return (1536, 864)
        elif aspect_ratio == "9:16":
            return (864, 1536)
        elif aspect_ratio == "4:3":
            return (1536, 1152)
        elif aspect_ratio == "3:4":
            return (1024, 1536)
        else:
            return (1024, 1024)


class QwenImageDiffSynthEliGenLoader:
    """
    Load EliGen (Entity-Level Image Generation) model for precise region control.
    Based on official DiffSynth-Studio EliGen implementation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eligen_model": (["DiffSynth-Studio/Qwen-Image-EliGen"], {
                    "default": "DiffSynth-Studio/Qwen-Image-EliGen",
                    "tooltip": "EliGen model for entity control"
                }),
                "enable_eligen": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable EliGen entity control"
                })
            }
        }

    RETURN_TYPES = ("QWEN_ELIGEN_CONFIG",)
    RETURN_NAMES = ("eligen_config",)
    FUNCTION = "load_eligen"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_eligen(self, eligen_model: str, enable_eligen: bool):
        """Load EliGen configuration for entity control."""
        try:
            eligen_config = {
                "eligen_model": eligen_model,
                "enable_eligen": enable_eligen,
                "model_path": f"models/{eligen_model}/model.safetensors",
                "loaded": False
            }

            logger.info(f"EliGen config created: {eligen_model}")
            return (eligen_config,)

        except Exception as e:
            logger.error(f"Failed to create EliGen config: {e}")
            raise RuntimeError(f"EliGen config creation failed: {e}")


class QwenImageDiffSynthEliGenEntityInput:
    """
    Create entity input for EliGen precise region control.
    Based on official EliGen examples with entity prompts and masks.

    Note on mask polarity:
    - ComfyUI convention (MASK/遮罩): black = active, white = inactive
    - EliGen (DiffSynth) expects: white = active, black = inactive
    This node provides an 'invert_mask' switch (default: True) to follow ComfyUI habit.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_prompt": ("STRING", {
                    "multiline": True,
                    "default": "魔法咖啡厅的宣传海报，主体是两杯魔法咖啡，背景是浅蓝色水雾",
                    "tooltip": "Global scene description"
                }),
                "entity_prompt_1": ("STRING", {
                    "multiline": True,
                    "default": "一杯红色魔法咖啡，杯中火焰燃烧",
                    "tooltip": "First entity description"
                }),
                "entity_mask_1": ("IMAGE", {
                    "tooltip": "Mask for first entity region"
                }),
            },
            "optional": {
                "entity_prompt_2": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Second entity description (optional)"
                }),
                "entity_mask_2": ("IMAGE", {
                    "tooltip": "Mask for second entity region (optional)"
                }),
                "entity_prompt_3": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Third entity description (optional)"
                }),
                "entity_mask_3": ("IMAGE", {
                    "tooltip": "Mask for third entity region (optional)"
                }),
                "entity_prompt_4": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Fourth entity description (optional)"
                }),
                "entity_mask_4": ("IMAGE", {
                    "tooltip": "Mask for fourth entity region (optional)"
                }),
                # 行为控制选项
                "auto_sort_by_position": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "按照顶部到下方、再从左到右对区域排序（对齐 QwenImage-Diffsynth）"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "输入若为 ComfyUI 遮罩（黑=有效），则需反相为 EliGen 期望的白=有效"
                }),
            }
        }

    RETURN_TYPES = ("ELIGEN_ARGS", "IMAGE")
    RETURN_NAMES = ("eligen_args", "mask")
    FUNCTION = "create_eligen_input"
    CATEGORY = "Ken-Chen/Qwen-Image/conditioning"

    def create_eligen_input(self, global_prompt: str, entity_prompt_1: str, entity_mask_1,
                           entity_prompt_2: str = "", entity_mask_2=None,
                           entity_prompt_3: str = "", entity_mask_3=None,
                           entity_prompt_4: str = "", entity_mask_4=None,
                           auto_sort_by_position: bool = True,
                           invert_mask: bool = True):
        """Create EliGen entity input following official API with optional mask inversion."""
        try:
            entity_prompts = []
            entity_masks = []

            def _expand_lines(prompt: str, mask):
                lines = [ln.strip() for ln in str(prompt).splitlines() if ln.strip()]
                for ln in lines:
                    entity_prompts.append(_sanitize_text_prompt(ln))
                    entity_masks.append(mask)

            # Add first entity (required)
            if isinstance(entity_prompt_1, str) and entity_prompt_1.strip():
                _expand_lines(entity_prompt_1, entity_mask_1)

            # Add optional entities
            for prompt, mask in [(entity_prompt_2, entity_mask_2),
                               (entity_prompt_3, entity_mask_3),
                               (entity_prompt_4, entity_mask_4)]:
                if isinstance(prompt, str) and prompt.strip() and mask is not None:
                    _expand_lines(prompt, mask)

            # 统一处理与排序：将遮罩转换为白=前景的 PIL，并按“从上到下、从左到右”排序
            pairs = [(p, m) for p, m in zip(entity_prompts, entity_masks) if m is not None]
            def _to_pil_binary(mask):
                if isinstance(mask, torch.Tensor):
                    arr = mask[0].detach().cpu().numpy()
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    if arr.max() <= 1.0 + 1e-6:
                        arr = arr * 255.0
                elif isinstance(mask, Image.Image):
                    arr = np.array(mask.convert('L'), dtype=np.float32)
                else:
                    arr = np.array(mask).astype(np.float32)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                bw = (arr > 127).astype(np.uint8) * 255
                if invert_mask:
                    bw = 255 - bw
                rgb = np.stack([bw, bw, bw], axis=-1)
                return Image.fromarray(rgb, mode='RGB')

            norm_pairs = []  # (prompt, pil_mask, bbox)
            for p, m in pairs:
                pil_mask = _to_pil_binary(m)
                bbox = pil_mask.getbbox()
                norm_pairs.append((p, pil_mask, bbox))

            if auto_sort_by_position:
                norm_pairs.sort(key=lambda x: (x[2][1] if x[2] else 1e9, x[2][0] if x[2] else 1e9))

            # 按排序结果回填用于下游 EliGen 的 prompts 与 masks
            entity_prompts = [pp for pp, _, _ in norm_pairs]
            entity_masks = [mm for _, mm, _ in norm_pairs]

            # 构建可视化预览：对标 QwenImage-Diffsynth 的 visualize_masks 实现
            try:
                # 半透明调色板（与参考实现一致，观感更好）
                colors = [
                    (165, 238, 173, 80), (76, 102, 221, 80), (221, 160, 77, 80), (204, 93, 71, 80),
                    (145, 187, 149, 80), (134, 141, 172, 80), (157, 137, 109, 80), (153, 104, 95, 80),
                    (165, 238, 173, 80), (76, 102, 221, 80), (221, 160, 77, 80), (204, 93, 71, 80),
                    (145, 187, 149, 80), (134, 141, 172, 80), (157, 137, 109, 80), (153, 104, 95, 80),
                ]

                valid_pairs = [(p, m) for p, m in zip(entity_prompts, entity_masks) if m is not None]
                preview_tensor = None
                if valid_pairs:
                    # 画布尺寸取第一张 mask
                    m0 = valid_pairs[0][1]
                    if isinstance(m0, torch.Tensor):
                        h, w = int(m0.shape[1]), int(m0.shape[2])
                    elif isinstance(m0, Image.Image):
                        w, h = m0.size
                    else:
                        arr0 = np.array(m0)
                        h, w = int(arr0.shape[0]), int(arr0.shape[1])

                    # 将输入的任意 IMAGE 统一为 0/255 的二值 PIL（白=前景区域，黑=背景）
                    def to_binary_pil(mask):
                        if isinstance(mask, torch.Tensor):
                            arr = mask[0].detach().cpu().numpy()
                            if arr.ndim == 3:
                                arr = arr[..., 0]
                            arr = arr.astype(np.float32)
                            if arr.max() <= 1.0 + 1e-6:
                                arr = arr * 255.0
                        elif isinstance(mask, Image.Image):
                            arr = np.array(mask.convert('L'), dtype=np.float32)
                        else:
                            arr = np.array(mask).astype(np.float32)
                            if arr.ndim == 3:
                                arr = arr[..., 0]
                        # 阈值二值化
                        bw = (arr > 127).astype(np.uint8) * 255
                        rgb = np.stack([bw, bw, bw], axis=-1)
                        return Image.fromarray(rgb, mode='RGB')


                    # 构建覆盖图层
                    base = Image.new('RGBA', (w, h), (0, 0, 0, 255))  # 黑色背景
                    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))

                    # 中文字体加载（优先使用 QwenImage-Diffsynth 同款字体）
                    from PIL import ImageDraw, ImageFont
                    def _load_cjk_font(size=28):
                        try:
                            base_dir = os.path.dirname(__file__)
                            qwen_font = os.path.abspath(os.path.join(base_dir, '..', 'QwenImage-Diffsynth', 'font', 'Arial-Unicode-Regular.ttf'))
                        except Exception:
                            qwen_font = None
                        candidates = [
                            qwen_font if qwen_font and os.path.exists(qwen_font) else None,
                            r"C:\\Windows\\Fonts\\msyh.ttc",
                            r"C:\\Windows\\Fonts\\msyhbd.ttc",
                            r"C:\\Windows\\Fonts\\simhei.ttf",
                            r"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                            r"/System/Library/Fonts/STHeiti Light.ttc",
                        ]
                        for fp in candidates:
                            if not fp:
                                continue
                            try:
                                return ImageFont.truetype(fp, size)
                            except Exception:
                                pass
                        return ImageFont.load_default()

                    font = _load_cjk_font(size=max(18, int(min(h, w) * 0.03)))

                    # 文本测量/换行工具，使长句在 mask 内自动换行
                    _measure_draw = ImageDraw.Draw(Image.new('RGB', (8, 8)))
                    def _text_size(t, f):
                        try:
                            x0, y0, x1, y1 = _measure_draw.textbbox((0, 0), t, font=f)
                            return x1 - x0, y1 - y0
                        except Exception:
                            return _measure_draw.textsize(t, font=f)

                    def wrap_text_to_width(t, f, max_width, max_lines=3):
                        if not t:
                            return []
                        lines, cur = [], ''
                        for ch in t:
                            w, _ = _text_size(cur + ch, f)
                            if w <= max_width:
                                cur += ch
                            else:
                                if cur:
                                    lines.append(cur)
                                cur = ch
                                if len(lines) >= max_lines - 1:
                                    # 剩余直接截断
                                    # 添加省略号
                                    while _text_size(cur + '…', f)[0] > max_width and len(cur) > 0:
                                        cur = cur[:-1]
                                    lines.append(cur + '…')
                                    return lines
                        if cur:
                            lines.append(cur)
                        return lines[:max_lines]

                    # 将相同 bbox 的多行实体进行分组，避免同一遮罩重复覆盖导致错位；在预览中按行对齐显示
                    groups = {}
                    for idx, (p, m) in enumerate(valid_pairs):
                        mp = to_binary_pil(m)
                        bb = mp.getbbox()
                        key = bb if bb is not None else ("none", idx)
                        if key not in groups:
                            groups[key] = {"mask_pil": mp, "prompts": []}
                        groups[key]["prompts"].append((p or '').strip())

                    for gi, (bb, g) in enumerate(groups.items()):
                        mask_pil = g["mask_pil"]
                        mask_rgba = mask_pil.convert('RGBA')
                        mask_data = mask_rgba.getdata()
                        color = colors[gi % len(colors)]
                        # 将白色像素替换为给定半透明颜色
                        new_data = [(color if px[:3] == (255, 255, 255) else (0, 0, 0, 0)) for px in mask_data]
                        mask_rgba.putdata(new_data)

                        # 在 mask 内侧按行堆叠绘制每条文案
                        draw = ImageDraw.Draw(mask_rgba)
                        bbox = mask_pil.getbbox()  # (x0,y0,x1,y1)
                        if bbox is not None:
                            x0, y0, x1, y1 = bbox
                            pad = 8
                            max_w = max(40, (x1 - x0) - pad * 2)
                            line_h = _text_size('测试', font)[1]
                            tx = x0 + pad
                            ty = y0 + pad
                            # 先计算背景宽高
                            max_line_w = 0
                            for t in g["prompts"]:
                                wrapped = wrap_text_to_width(t, font, max_w, max_lines=1)
                                ln = wrapped[0] if wrapped else ''
                                w = _text_size(ln, font)[0]
                                if w > max_line_w:
                                    max_line_w = w
                            bg_w = max_line_w + pad * 2
                            bg_h = line_h * len(g["prompts"]) + pad
                            draw.rectangle([tx - pad, ty - pad, tx - pad + bg_w, ty - pad + bg_h], fill=(0,0,0,128))

                            for j, text in enumerate(g["prompts"]):
                                wrapped = wrap_text_to_width(text, font, max_w, max_lines=1)
                                line = wrapped[0] if wrapped else ''
                                y = ty + j * line_h
                                try:
                                    draw.text((tx, y), line, fill=(255, 255, 255, 255), font=font)
                                except Exception:
                                    draw.text((tx, y), line, fill=(255, 255, 255, 255))

                        overlay = Image.alpha_composite(overlay, mask_rgba)

                    # 合成结果
                    preview_pil = Image.alpha_composite(base, overlay).convert('RGB')
                    preview_tensor = pil2comfy(preview_pil)
                else:
                    preview_pil = Image.new('RGB', (512, 512), 'black')
                    preview_tensor = pil2comfy(preview_pil)
            except Exception as _e:
                logger.warning(f"Failed to build EliGen mask preview (QwenImage style): {_e}")
                preview_pil = Image.new('RGB', (512, 512), 'black')
                preview_tensor = pil2comfy(preview_pil)

            eligen_args = {
                "global_prompt": global_prompt,
                "prompts": entity_prompts,
                "masks": entity_masks,
                "entity_count": len(entity_prompts)
            }

            logger.info(f"Created EliGen input with {len(entity_prompts)} entities")
            return (eligen_args, preview_tensor)

        except Exception as e:
            logger.error(f"Failed to create EliGen input: {e}")
            raise RuntimeError(f"EliGen input creation failed: {e}")


class QwenImageDiffSynthDistillPipelineLoader:
    """
    Dedicated loader for Qwen-Image Distill model (DiffSynth-Studio/Qwen-Image-Distill-Full).
    Also supports loading a Distill-LoRA on top.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_optimization": (VRAM_OPTIMIZATION_OPTS, {
                    "default": "HighRAM_LowVRAM",
                    "tooltip": "VRAM optimization strategy"
                }),
                "offload_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload models to CPU when not in use"
                }),
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision"
                }),
            },
            "optional": {
                "distill_lora": (["none", "None", "NONE", "null", "Null", "NULL"] + folder_paths.get_filename_list("loras"), {
                    "default": "none",
                    "tooltip": "Optional Distill-LoRA from loras folder"
                }),
                "external_lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional external LoRA path (e.g., models/DiffSynth-Studio/Qwen-Image-Distill-LoRA/model.safetensors)"
                }),
                "download_distill_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Online download Distill-LoRA via ModelScope"
                }),
                "modelscope_model_id": ("STRING", {
                    "default": "DiffSynth-Studio/Qwen-Image-Distill-LoRA",
                    "multiline": False,
                    "tooltip": "ModelScope model id for Distill-LoRA"
                }),
                "download_dir": ("STRING", {
                    "default": "models/DiffSynth-Studio/Qwen-Image-Distill-LoRA",
                    "multiline": False,
                    "tooltip": "Local directory to store downloaded Distill-LoRA"
                }),
                "allow_file_pattern": ("STRING", {
                    "default": "model.safetensors",
                    "multiline": False,
                    "tooltip": "File name or pattern to allow when downloading"
                }),
                # Online EliGen-LoRA download options (optional)
                "download_eligen_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Online download EliGen LoRA via ModelScope"
                }),
                "modelscope_model_id_eligen": ("STRING", {
                    "default": "DiffSynth-Studio/Qwen-Image-EliGen",
                    "multiline": False,
                    "tooltip": "ModelScope model id for EliGen LoRA"
                }),
                "download_dir_eligen": ("STRING", {
                    "default": "models/DiffSynth-Studio/Qwen-Image-EliGen",
                    "multiline": False,
                    "tooltip": "Local directory to store downloaded EliGen LoRA"
                }),
                "allow_file_pattern_eligen": ("STRING", {
                    "default": "model.safetensors",
                    "multiline": False,
                    "tooltip": "File name or pattern to allow when downloading EliGen LoRA"
                }),
                # LoRA load behavior controls (Distill Loader)
                "lora_source_policy": (["auto_accumulate", "download_only", "local_only"], {
                    "default": "auto_accumulate",
                    "tooltip": "How to apply LoRAs when both downloaded and local are available"
                }),
                "distill_lora_download_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only download Distill-LoRA, do not auto-load"
                }),
                "eligen_lora_download_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only download EliGen LoRA, do not auto-load"
                }),
            }
        }

    RETURN_TYPES = ("QWEN_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_distill_pipeline"
    CATEGORY = "Ken-Chen/Qwen-Image/loaders"

    def load_distill_pipeline(self, vram_optimization: str, offload_to_cpu: bool, torch_dtype: str,
                              distill_lora: str = "none", external_lora_path: str = "",
                              download_distill_lora: bool = False, modelscope_model_id: str = "DiffSynth-Studio/Qwen-Image-Distill-LoRA",
                              download_dir: str = "models/DiffSynth-Studio/Qwen-Image-Distill-LoRA", allow_file_pattern: str = "model.safetensors",
                              download_eligen_lora: bool = False, modelscope_model_id_eligen: str = "DiffSynth-Studio/Qwen-Image-EliGen",
                              download_dir_eligen: str = "models/DiffSynth-Studio/Qwen-Image-EliGen", allow_file_pattern_eligen: str = "model.safetensors",
                              lora_source_policy: str = "auto_accumulate",
                              distill_lora_download_only: bool = False,
                              eligen_lora_download_only: bool = False):
        try:
            # Standard dtype handling
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if torch_dtype not in dtype_map:
                raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

            pipeline_dtype = dtype_map[torch_dtype]
            offload_dtype = None
            logger.info(f"Using dtype: {torch_dtype}")

            available, QwenImagePipeline, ModelConfig, ControlNetInput, ModelManager = _get_diffsynth_classes()
            if not available:
                raise RuntimeError("DiffSynth-Studio is not available in this environment")

            # Build model_configs for Distill-Full + original TE/VAE
            # 目标：offload_to_cpu=True 时也能达到≈70% 显存占用与较高速度。
            # 策略：始终只对 transformer（DiT）做 offload；text_encoder/vae 常驻 GPU。
            # 这样避免频繁 CPU↔GPU 传输，同时利用 MMGP 对主干做流式管理。
            off_dev_transformer = ("cpu" if offload_to_cpu else "cuda")
            off_dev_te_vae = "cuda"  # 始终让 TE/VAE 在 GPU

            model_configs = [
                ModelConfig(
                    model_id="DiffSynth-Studio/Qwen-Image-Distill-Full",
                    origin_file_pattern="diffusion_pytorch_model*.safetensors",
                    offload_device=off_dev_transformer,
                    offload_dtype=offload_dtype
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="text_encoder/model*.safetensors",
                    offload_device=off_dev_te_vae,
                    offload_dtype=offload_dtype
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    offload_device=off_dev_te_vae,
                    offload_dtype=offload_dtype
                ),
            ]
            tokenizer_config = ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="tokenizer/",
                offload_device=off_dev_te_vae
            )

            pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=pipeline_dtype,  # Use pipeline_dtype (bfloat16 for FP8, original for others)
                device="cuda",
                model_configs=model_configs,
                tokenizer_config=tokenizer_config,
            )
            try:
                setattr(pipe, "_augment_is_distill", True)
            except Exception:
                pass

            # Apply VRAM optimization profile - 仅对 transformer 使用 MMGP；统一 70% workingVRAM
            if vram_optimization == 'No_Optimization':
                pipe.enable_vram_management()
                logger.info("Enabled pipeline built-in VRAM management")
            else:
                try:
                    if MMGP_AVAILABLE and offload_to_cpu:
                        offload_pipe = {"transformer": pipe.dit}  # 仅管理主干
                        if vram_optimization == 'HighRAM_HighVRAM':
                            opt = mmgp_profile_type.HighRAM_HighVRAM
                        elif vram_optimization == 'HighRAM_LowVRAM':
                            opt = mmgp_profile_type.HighRAM_LowVRAM
                        elif vram_optimization == 'LowRAM_HighVRAM':
                            opt = mmgp_profile_type.LowRAM_HighVRAM
                        elif vram_optimization == 'LowRAM_LowVRAM':
                            opt = mmgp_profile_type.LowRAM_LowVRAM
                        elif vram_optimization == 'VerylowRAM_LowVRAM':
                            opt = mmgp_profile_type.VerylowRAM_LowVRAM
                        else:
                            opt = mmgp_profile_type.HighRAM_LowVRAM
                        try:
                            mmgp_offload.profile(
                                offload_pipe,
                                opt,
                                verboseLevel=1,
                                pinnedMemory=True,
                                asyncTransfers=True,
                                budgets={"transformer": "90%"},
                                workingVRAM="70%",
                            )
                        except TypeError:
                            mmgp_offload.profile(offload_pipe, opt, verboseLevel=1)
                        logger.info(f"Applied mmgp offload profile: {vram_optimization} (TE/VAE on GPU, transformer offloaded, workingVRAM=70%)")
                    else:
                        pipe.enable_vram_management()
                        logger.info("Using pipeline built-in VRAM management (MMGP unavailable or offload_to_cpu=False)")
                except Exception as _e:
                    logger.warning(f"VRAM optimization failed: {_e}")
                    pipe.enable_vram_management()
                    logger.info("Fallback: Enabled pipeline built-in VRAM management")

            # Online download Distill-LoRA if requested
            if download_distill_lora:
                try:
                    from modelscope import snapshot_download
                    snapshot_download(modelscope_model_id, local_dir=download_dir, allow_file_pattern=allow_file_pattern)
                    # If a modelscope path was provided, use it as external lora as default
                    default_downloaded_path = os.path.join(download_dir, allow_file_pattern)
                    if os.path.exists(default_downloaded_path):
                        external_lora_path = default_downloaded_path
                        logger.info(f"Downloaded Distill-LoRA to: {external_lora_path}")
                except Exception as e:
                    logger.warning(f"ModelScope download failed: {e}")

            # Online download EliGen LoRA if requested
            if download_eligen_lora:
                try:
                    from modelscope import snapshot_download as snapshot_download_eligen
                    snapshot_download_eligen(modelscope_model_id_eligen, local_dir=download_dir_eligen, allow_file_pattern=allow_file_pattern_eligen)
                    default_eligen_path = os.path.join(download_dir_eligen, allow_file_pattern_eligen)
                    if os.path.exists(default_eligen_path):
                        if not eligen_lora_download_only and lora_source_policy != "download_only":
                            # load EliGen lora as well
                            pipe.load_lora(pipe.dit, default_eligen_path)
                            logger.info(f"Downloaded EliGen-LoRA and loaded: {default_eligen_path}")
                except Exception as e:
                    logger.warning(f"ModelScope (EliGen) download failed: {e}")

            # Load optional Distill-LoRA(s)
            lora_paths = []
            if not _is_none_choice(distill_lora):
                try:
                    lora_paths.append(folder_paths.get_full_path("loras", distill_lora))
                except Exception:
                    pass
            if external_lora_path and external_lora_path.strip():
                lora_paths.append(external_lora_path.strip())

            for lp in lora_paths:
                try:
                    if lora_source_policy != "download_only":
                        pipe.load_lora(pipe.dit, lp)
                        try:
                            if "lightning" in os.path.basename(lp).lower():
                                setattr(pipe, "_augment_has_lightning", True)
                                logger.info("Detected Lightning LoRA (distill loader); flag set on pipeline")
                        except Exception:
                            pass
                        logger.info(f"Loaded Distill LoRA: {lp}")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA {lp}: {e}")

            logger.info("Distill pipeline loaded successfully")
            return (pipe,)
        except Exception as e:
            logger.error(f"Failed to load distill pipeline: {e}")
            raise

# Node mappings (with environment-based disable switch for A/B isolation)
_DISABLE_THIS_PACKAGE = os.environ.get("COMFYUI_AUGMENT_DISABLE_QWEN_IMAGE", "0") in ("1", "true", "True")

if not _DISABLE_THIS_PACKAGE:
    NODE_CLASS_MAPPINGS = {
        "QwenImageDiffSynthLoRALoader": QwenImageDiffSynthLoRALoader,
        "QwenImageDiffSynthLoRAMulti": QwenImageDiffSynthLoRAMulti,
        "QwenImageDiffSynthControlNetLoader": QwenImageDiffSynthControlNetLoader,
        "QwenImageDiffSynthPipelineLoader": QwenImageDiffSynthPipelineLoader,
        "QwenImageDiffSynthControlNetInput": QwenImageDiffSynthControlNetInput,
        "QwenDiffSynthSetControlArgs": QwenDiffSynthSetControlArgs,
        "QwenImageDiffSynthAdvancedSampler": QwenImageDiffSynthAdvancedSampler,
        "QwenImageDiffSynthMemoryManager": QwenImageDiffSynthMemoryManager,
        "QwenVRAMCleanupPassThrough": QwenVRAMCleanupPassThrough,
        "QwenMemoryManagerPassThrough": QwenMemoryManagerPassThrough,
        "QwenImageDiffSynthRatio2Size": QwenImageDiffSynthRatio2Size,
        "QwenImageDiffSynthEliGenLoader": QwenImageDiffSynthEliGenLoader,
        "QwenImageDiffSynthEliGenEntityInput": QwenImageDiffSynthEliGenEntityInput,
        "QwenImageDiffSynthDistillPipelineLoader": QwenImageDiffSynthDistillPipelineLoader,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "QwenImageDiffSynthLoRALoader": "🎨 Qwen DiffSynth LoRA Loader",
        "QwenImageDiffSynthLoRAMulti": "🎨 Qwen DiffSynth LoRA Multi",
        "QwenImageDiffSynthControlNetLoader": "🎨 Qwen DiffSynth ControlNet Loader",
        "QwenImageDiffSynthPipelineLoader": "🎨 Qwen DiffSynth Pipeline Loader",
        "QwenImageDiffSynthControlNetInput": "🎨 Qwen DiffSynth ControlNet Input",
        "QwenDiffSynthSetControlArgs": "🎨 Qwen DiffSynth Set Control Args",
        "QwenImageDiffSynthAdvancedSampler": "🎨 Qwen DiffSynth Advanced Sampler",
        "QwenImageDiffSynthMemoryManager": "🎨 Qwen DiffSynth Memory Manager",
        "QwenVRAMCleanupPassThrough": "📌 VRAM-Cleanup (Pass-Through)",
        "QwenMemoryManagerPassThrough": "📌 Memory Manager (Pass-Through)",
        "QwenImageDiffSynthRatio2Size": "🎨 Qwen DiffSynth Ratio to Size",
        "QwenImageDiffSynthEliGenLoader": "🎨 Qwen DiffSynth EliGen Loader",
        "QwenImageDiffSynthEliGenEntityInput": "🎨 Qwen DiffSynth EliGen Entity Input",
        "QwenImageDiffSynthDistillPipelineLoader": "🎨 Qwen DiffSynth Distill Pipeline Loader",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
