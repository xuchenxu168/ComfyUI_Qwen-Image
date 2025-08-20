"""
Utility functions for Qwen-Image ComfyUI plugin.

This module provides helper functions for image processing, model management,
and other common operations used across the plugin.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image."""
    image_array = tensor.squeeze().cpu().numpy()
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Handle different tensor shapes
    if len(image_array.shape) == 3:
        if image_array.shape[0] == 3:  # CHW format
            image_array = np.transpose(image_array, (1, 2, 0))
        # HWC format is already correct
    elif len(image_array.shape) == 2:  # Grayscale
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {image_array.shape}")
    
    return Image.fromarray(image_array)

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor format."""
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Ensure RGB format
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 4:  # RGBA
        # Convert RGBA to RGB by compositing over white background
        alpha = image_array[:, :, 3:4]
        rgb = image_array[:, :, :3]
        image_array = rgb * alpha + (1 - alpha)
    
    return torch.from_numpy(image_array)[None,]

def detect_language(text: str) -> str:
    """Simple language detection for Chinese vs English."""
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    # Count both alphabetic characters and Chinese characters
    english_chars = sum(1 for char in text if char.isalpha() and not ('\u4e00' <= char <= '\u9fff'))
    total_chars = chinese_chars + english_chars

    if total_chars == 0:
        return "en"

    chinese_ratio = chinese_chars / total_chars
    return "zh" if chinese_ratio > 0.2 else "en"

def enhance_prompt_for_text_rendering(text: str, style: str = "elegant", 
                                    color: str = "black", size: str = "medium") -> str:
    """Enhance prompt specifically for high-quality text rendering."""
    language = detect_language(text)
    
    base_enhancements = [
        "high-quality typography",
        "crisp and clear text",
        "perfect character formation",
        "professional text rendering",
        "ultra-sharp text details"
    ]
    
    if language == "zh":
        base_enhancements.extend([
            "beautiful Chinese calligraphy",
            "perfect Chinese character strokes",
            "traditional Chinese typography",
            "elegant Chinese font"
        ])
    
    style_enhancements = {
        "elegant": ["refined typography", "sophisticated font design"],
        "modern": ["contemporary font", "clean modern typography"],
        "traditional": ["classic typography", "traditional font style"],
        "artistic": ["creative typography", "artistic font design"],
        "calligraphy": ["calligraphic style", "handwritten elegance"]
    }
    
    size_enhancements = {
        "small": ["fine text details"],
        "medium": ["well-proportioned text"],
        "large": ["bold prominent text"],
        "extra_large": ["large display text", "headline typography"]
    }
    
    enhancements = base_enhancements.copy()
    enhancements.extend(style_enhancements.get(style, []))
    enhancements.extend(size_enhancements.get(size, []))
    
    return ", ".join(enhancements)

def create_text_overlay(image: Image.Image, text: str, position: str = "center",
                       color: str = "white", font_size: int = 48) -> Image.Image:
    """Create a text overlay on an image (fallback method)."""
    try:
        # Create a copy of the image
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Try to load a font, fallback to default if not available
        try:
            # Try to find a suitable font for Chinese text
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/Windows/Fonts/msyh.ttc",  # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            
            if font is None:
                font = ImageFont.load_default()
                
        except Exception:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        img_width, img_height = img_with_text.size
        
        position_map = {
            "center": ((img_width - text_width) // 2, (img_height - text_height) // 2),
            "top": ((img_width - text_width) // 2, 50),
            "bottom": ((img_width - text_width) // 2, img_height - text_height - 50),
            "left": (50, (img_height - text_height) // 2),
            "right": (img_width - text_width - 50, (img_height - text_height) // 2)
        }
        
        text_position = position_map.get(position, position_map["center"])
        
        # Draw text with outline for better visibility
        outline_color = "black" if color == "white" else "white"
        for adj in range(-2, 3):
            for adj2 in range(-2, 3):
                draw.text((text_position[0] + adj, text_position[1] + adj2), 
                         text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text(text_position, text, font=font, fill=color)
        
        return img_with_text
        
    except Exception as e:
        logger.warning(f"Failed to create text overlay: {e}")
        return image

def validate_aspect_ratio(width: int, height: int) -> Tuple[int, int]:
    """Validate and adjust aspect ratio to supported values."""
    # Ensure dimensions are multiples of 8 (common requirement for diffusion models)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    # Ensure minimum dimensions
    width = max(width, 512)
    height = max(height, 512)
    
    # Ensure maximum dimensions
    width = min(width, 2048)
    height = min(height, 2048)
    
    return width, height

def download_model_if_needed(model_id: str, cache_dir: Optional[str] = None) -> bool:
    """Check if model exists locally, download if needed."""
    try:
        from huggingface_hub import snapshot_download
        
        # Try to download/verify model
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        return False

def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_info": {}
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_cached = torch.cuda.memory_reserved(i)
            
            info["memory_info"][f"cuda:{i}"] = {
                "name": props.name,
                "total_memory": memory_total,
                "allocated_memory": memory_allocated,
                "cached_memory": memory_cached,
                "free_memory": memory_total - memory_allocated
            }
    
    return info

def log_generation_params(params: Dict[str, Any]) -> None:
    """Log generation parameters for debugging."""
    logger.info("Generation Parameters:")
    for key, value in params.items():
        if isinstance(value, str) and len(value) > 100:
            logger.info(f"  {key}: {value[:100]}...")
        else:
            logger.info(f"  {key}: {value}")

class ModelCache:
    """Simple model cache to avoid reloading models."""
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, model: Any) -> None:
        """Store model in cache."""
        self._cache[key] = model
    
    def clear(self) -> None:
        """Clear all cached models."""
        self._cache.clear()
    
    def remove(self, key: str) -> None:
        """Remove specific model from cache."""
        if key in self._cache:
            del self._cache[key]

# Global model cache instance
model_cache = ModelCache()
