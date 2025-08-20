"""
Configuration management for ComfyUI Qwen-Image Plugin

This module handles configuration settings, model paths, and user preferences
for the Qwen-Image ComfyUI plugin.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class QwenImageConfig:
    """Configuration manager for Qwen-Image plugin."""
    
    def __init__(self):
        self.plugin_dir = Path(__file__).parent
        self.config_file = self.plugin_dir / "config.json"
        self.default_config = self._get_default_config()
        self.config = self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            "model": {
                "default_model_id": "Qwen/Qwen-Image",
                "cache_dir": "",
                "torch_dtype": "auto",
                "device": "auto",
                "use_safetensors": True,
                "enable_model_cache": True,
                "max_cached_models": 2
            },
            "generation": {
                "default_aspect_ratio": "16:9",
                "default_steps": 50,
                "default_cfg_scale": 4.0,
                "enable_prompt_enhancement": True,
                "max_prompt_length": 1000,
                "default_language": "auto"
            },
            "text_rendering": {
                "default_style": "elegant",
                "default_color": "black",
                "default_size": "medium",
                "default_position": "center",
                "enable_chinese_optimization": True,
                "fallback_font_rendering": True
            },
            "editing": {
                "default_strength": 0.8,
                "default_edit_steps": 50,
                "default_guidance_scale": 7.5,
                "enable_mask_preprocessing": True
            },
            "performance": {
                "enable_memory_optimization": True,
                "enable_attention_slicing": False,
                "enable_cpu_offload": False,
                "batch_size": 1,
                "enable_xformers": False
            },
            "ui": {
                "show_advanced_options": False,
                "enable_progress_bar": True,
                "auto_preview": True,
                "save_generation_params": True
            },
            "logging": {
                "log_level": "INFO",
                "enable_generation_logging": True,
                "log_file": "qwen_image.log"
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.default_config, config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
                logger.info("Using default configuration")
        
        # Create default config file
        self.save_config(self.default_config)
        return self.default_config.copy()
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file."""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.default_model_id')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        try:
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            return self.save_config()
        except Exception as e:
            logger.error(f"Failed to set config value {key_path}: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults."""
        self.config = self.default_config.copy()
        return self.save_config()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.config.get("model", {})
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation-specific configuration."""
        return self.config.get("generation", {})
    
    def get_text_rendering_config(self) -> Dict[str, Any]:
        """Get text rendering configuration."""
        return self.config.get("text_rendering", {})
    
    def get_editing_config(self) -> Dict[str, Any]:
        """Get editing configuration."""
        return self.config.get("editing", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config.get("performance", {})
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            "QWEN_IMAGE_MODEL_ID": "model.default_model_id",
            "QWEN_IMAGE_CACHE_DIR": "model.cache_dir",
            "QWEN_IMAGE_DEVICE": "model.device",
            "QWEN_IMAGE_DTYPE": "model.torch_dtype",
            "QWEN_IMAGE_LOG_LEVEL": "logging.log_level",
            "QWEN_IMAGE_ENABLE_XFORMERS": "performance.enable_xformers",
            "QWEN_IMAGE_MEMORY_OPTIMIZATION": "performance.enable_memory_optimization"
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_value.lower() in ('true', 'false'):
                    env_value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif env_value.replace('.', '').isdigit():
                    env_value = float(env_value)
                
                self.set(config_path, env_value)
                logger.info(f"Updated {config_path} from environment: {env_value}")

# Global configuration instance
config = QwenImageConfig()

# Load environment variables on import
config.update_from_env()

# Convenience functions
def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value."""
    return config.get(key_path, default)

def set_config(key_path: str, value: Any) -> bool:
    """Set configuration value."""
    return config.set(key_path, value)

def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return config.get_model_config()

def get_generation_config() -> Dict[str, Any]:
    """Get generation configuration."""
    return config.get_generation_config()

def get_text_rendering_config() -> Dict[str, Any]:
    """Get text rendering configuration."""
    return config.get_text_rendering_config()

def get_editing_config() -> Dict[str, Any]:
    """Get editing configuration."""
    return config.get_editing_config()

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    return config.get_performance_config()
