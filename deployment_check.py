#!/usr/bin/env python3
"""
Deployment Check Script for ComfyUI Qwen-Image Plugin

This script performs comprehensive checks to ensure the plugin is ready for deployment
and will work correctly in a ComfyUI environment.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_structure() -> bool:
    """Check that all required files are present."""
    logger.info("Checking file structure...")
    
    required_files = [
        "__init__.py",
        "qwen_image_nodes.py", 
        "utils.py",
        "config.py",
        "requirements.txt",
        "pyproject.toml",
        "install.py",
        "test_plugin.py",
        "README.md",
        "USAGE_GUIDE.md",
        "CHANGELOG.md",
        "LICENSE",
        "PLUGIN_SUMMARY.md"
    ]
    
    required_dirs = [
        "example_workflows"
    ]
    
    plugin_dir = Path(__file__).parent
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not (plugin_dir / file).exists():
            missing_files.append(file)
    
    for dir_name in required_dirs:
        if not (plugin_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_files or missing_dirs:
        logger.error(f"❌ Missing files: {missing_files}")
        logger.error(f"❌ Missing directories: {missing_dirs}")
        return False
    
    logger.info("✓ All required files and directories present")
    return True

def check_example_workflows() -> bool:
    """Check that example workflows are present and valid."""
    logger.info("Checking example workflows...")
    
    workflows_dir = Path(__file__).parent / "example_workflows"
    expected_workflows = [
        "basic_text_generation.json",
        "chinese_calligraphy.json", 
        "bilingual_signage.json",
        "advanced_image_editing.json",
        "logo_design_workflow.json"
    ]
    
    for workflow in expected_workflows:
        workflow_path = workflows_dir / workflow
        if not workflow_path.exists():
            logger.error(f"❌ Missing workflow: {workflow}")
            return False
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            # Basic validation
            if "nodes" not in workflow_data or "links" not in workflow_data:
                logger.error(f"❌ Invalid workflow structure: {workflow}")
                return False
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in workflow {workflow}: {e}")
            return False
    
    logger.info("✓ All example workflows present and valid")
    return True

def check_dependencies() -> bool:
    """Check that all dependencies are properly specified."""
    logger.info("Checking dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    pyproject_file = Path(__file__).parent / "pyproject.toml"
    
    # Check requirements.txt
    if not requirements_file.exists():
        logger.error("❌ requirements.txt not found")
        return False
    
    with open(requirements_file, 'r') as f:
        requirements = f.read().strip().split('\n')
    
    required_packages = [
        'torch', 'transformers', 'diffusers', 'accelerate', 
        'safetensors', 'Pillow', 'opencv-python', 'numpy',
        'huggingface-hub', 'datasets', 'tqdm', 'requests'
    ]
    
    for package in required_packages:
        if not any(package in req for req in requirements if req.strip() and not req.startswith('#')):
            logger.error(f"❌ Missing required package in requirements.txt: {package}")
            return False
    
    logger.info("✓ Dependencies properly specified")
    return True

def check_node_definitions() -> bool:
    """Check that all nodes are properly defined."""
    logger.info("Checking node definitions...")

    try:
        # Mock ComfyUI dependencies for testing
        import sys
        sys.modules['folder_paths'] = type('MockModule', (), {})()
        sys.modules['comfy.model_management'] = type('MockModule', (), {
            'get_torch_device': lambda: 'cpu'
        })()

        # Import nodes
        sys.path.insert(0, str(Path(__file__).parent))
        from qwen_image_nodes import (
            QwenImageModelLoader,
            QwenImageGenerate,
            QwenImageTextRender,
            QwenImageEdit,
            QwenImageUnderstanding
        )
        
        nodes = [
            QwenImageModelLoader,
            QwenImageGenerate,
            QwenImageTextRender,
            QwenImageEdit,
            QwenImageUnderstanding
        ]
        
        for node_class in nodes:
            # Check required methods
            if not hasattr(node_class, 'INPUT_TYPES'):
                logger.error(f"❌ {node_class.__name__} missing INPUT_TYPES")
                return False
            
            if not hasattr(node_class, 'RETURN_TYPES'):
                logger.error(f"❌ {node_class.__name__} missing RETURN_TYPES")
                return False
            
            if not hasattr(node_class, 'FUNCTION'):
                logger.error(f"❌ {node_class.__name__} missing FUNCTION")
                return False
            
            if not hasattr(node_class, 'CATEGORY'):
                logger.error(f"❌ {node_class.__name__} missing CATEGORY")
                return False
            
            # Check INPUT_TYPES is callable and returns dict
            try:
                input_types = node_class.INPUT_TYPES()
                if not isinstance(input_types, dict):
                    logger.error(f"❌ {node_class.__name__}.INPUT_TYPES() must return dict")
                    return False
                
                if 'required' not in input_types:
                    logger.error(f"❌ {node_class.__name__}.INPUT_TYPES() must have 'required' key")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error calling {node_class.__name__}.INPUT_TYPES(): {e}")
                return False
        
        logger.info("✓ All nodes properly defined")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import nodes: {e}")
        return False

def check_documentation() -> bool:
    """Check that documentation is complete."""
    logger.info("Checking documentation...")
    
    docs = {
        "README.md": ["Installation", "Quick Start", "Features"],
        "USAGE_GUIDE.md": ["Node Reference", "Chinese Text", "Examples"],
        "CHANGELOG.md": ["[1.0.0]", "Added", "Features"]
    }
    
    plugin_dir = Path(__file__).parent
    
    for doc_file, required_sections in docs.items():
        doc_path = plugin_dir / doc_file
        if not doc_path.exists():
            logger.error(f"❌ Missing documentation: {doc_file}")
            return False
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for section in required_sections:
            if section not in content:
                logger.error(f"❌ Missing section '{section}' in {doc_file}")
                return False
    
    logger.info("✓ Documentation complete")
    return True

def check_configuration() -> bool:
    """Check that configuration system works."""
    logger.info("Checking configuration system...")
    
    try:
        from config import get_config, set_config
        
        # Test basic config operations
        model_id = get_config("model.default_model_id")
        if model_id != "Qwen/Qwen-Image":
            logger.error(f"❌ Unexpected default model ID: {model_id}")
            return False
        
        # Test setting and getting
        test_key = "test.deployment_check"
        test_value = "deployment_test_value"
        
        if not set_config(test_key, test_value):
            logger.error("❌ Failed to set config value")
            return False
        
        retrieved_value = get_config(test_key)
        if retrieved_value != test_value:
            logger.error(f"❌ Config value mismatch: expected {test_value}, got {retrieved_value}")
            return False
        
        logger.info("✓ Configuration system working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration system error: {e}")
        return False

def generate_deployment_report() -> Dict:
    """Generate a comprehensive deployment report."""
    logger.info("Generating deployment report...")
    
    checks = [
        ("File Structure", check_file_structure),
        ("Example Workflows", check_example_workflows),
        ("Dependencies", check_dependencies),
        ("Node Definitions", check_node_definitions),
        ("Documentation", check_documentation),
        ("Configuration", check_configuration)
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = {"passed": result, "error": None}
            if not result:
                all_passed = False
        except Exception as e:
            results[check_name] = {"passed": False, "error": str(e)}
            all_passed = False
            logger.error(f"❌ {check_name} check failed with exception: {e}")
    
    # Generate summary
    report = {
        "timestamp": str(Path(__file__).parent / "deployment_check.py"),
        "overall_status": "READY" if all_passed else "NOT_READY",
        "checks": results,
        "summary": {
            "total_checks": len(checks),
            "passed": sum(1 for r in results.values() if r["passed"]),
            "failed": sum(1 for r in results.values() if not r["passed"])
        }
    }
    
    return report

def main():
    """Main deployment check function."""
    logger.info("="*60)
    logger.info("ComfyUI Qwen-Image Plugin Deployment Check")
    logger.info("="*60)
    
    report = generate_deployment_report()
    
    logger.info("\n" + "="*60)
    logger.info("DEPLOYMENT REPORT")
    logger.info("="*60)
    
    for check_name, result in report["checks"].items():
        status = "✓ PASS" if result["passed"] else "❌ FAIL"
        logger.info(f"{status} {check_name}")
        if result["error"]:
            logger.info(f"    Error: {result['error']}")
    
    logger.info(f"\nSummary: {report['summary']['passed']}/{report['summary']['total_checks']} checks passed")
    logger.info(f"Overall Status: {report['overall_status']}")
    
    if report["overall_status"] == "READY":
        logger.info("\n🎉 Plugin is ready for deployment!")
        logger.info("You can now:")
        logger.info("1. Copy the plugin to ComfyUI/custom_nodes/")
        logger.info("2. Install dependencies: pip install -r requirements.txt")
        logger.info("3. Restart ComfyUI")
        logger.info("4. Look for 🎨 Qwen-Image nodes in the node menu")
    else:
        logger.warning("\n⚠ Plugin is not ready for deployment.")
        logger.warning("Please fix the issues above before deploying.")
    
    return report["overall_status"] == "READY"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
