#!/usr/bin/env python3
"""
Syntax check for ComfyUI Qwen-Image Plugin v2.0
Validates Python syntax without importing ComfyUI dependencies.
"""

import ast
import os
import sys

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    """Check syntax of all Python files in the plugin."""
    print("🔍 ComfyUI Qwen-Image Plugin v2.0 Syntax Check")
    print("=" * 50)
    
    # Files to check
    files_to_check = [
        "__init__.py",
        "qwen_image_nodes.py",
        "utils.py",
        "config.py",
        "test_new_architecture.py"
    ]
    
    passed = 0
    total = len(files_to_check)
    
    for file_name in files_to_check:
        if os.path.exists(file_name):
            print(f"📄 Checking {file_name}...")
            is_valid, error = check_python_syntax(file_name)
            
            if is_valid:
                print(f"✅ {file_name}: Syntax OK")
                passed += 1
            else:
                print(f"❌ {file_name}: {error}")
        else:
            print(f"⚠️  {file_name}: File not found")
    
    print("\n" + "=" * 50)
    print(f"📊 Syntax Check Results: {passed}/{total} files passed")
    
    if passed == total:
        print("🎉 All files have valid Python syntax!")
        return True
    else:
        print("⚠️  Some files have syntax errors.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
