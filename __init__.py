"""
ComfyUI Qwen-Image Plugin

A comprehensive ComfyUI plugin for Qwen-Image integration using ComfyUI's standard model loading architecture.

This init is designed to be SAFE and NON-INTRUSIVE:
- Never raises on import; falls back to empty mappings on any error
- Supports environment switch COMFYUI_AUGMENT_DISABLE_QWEN_IMAGE=1 to disable registration entirely
"""

import os
import logging

logger = logging.getLogger(__name__)

# Environment switch: when set, do not register any nodes from this package
_DISABLE_THIS_PACKAGE = os.environ.get("COMFYUI_AUGMENT_DISABLE_QWEN_IMAGE", "0") in ("1", "true", "True")

if not _DISABLE_THIS_PACKAGE:
    try:
        from .qwen_image_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except Exception as e:
        # Fail-safe: never break other extensions' loading
        logger.warning(f"ComfyUI_Qwen-Image failed to load nodes: {e}")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Plugin metadata
WEB_DIRECTORY = "./web"
__version__ = "2.2.0"
__author__ = "Augment Agent"
__description__ = "ComfyUI plugin for Qwen-Image with advanced diffusion loader, separated model loading architecture and exceptional Chinese text rendering"
