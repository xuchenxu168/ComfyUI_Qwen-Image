# ComfyUI Qwen-Image + DiffSynth (Enhanced README)

This README consolidates and updates all features in this project, including the latest DiffSynth integration, Distill behavior, VRAM strategies, and Blockwise ControlNet Inpaint support.

## Highlights (What’s New)
- Inpaint ControlNet: Full support for DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint
- Smarter sampler rules
  - Distill models: never auto-increase steps due to enhanced_quality; cap steps at ≤15 only if user sets >15
  - Lightning LoRA: respect user steps (e.g., 4/8), enhanced_quality only boosts CFG a bit
- VRAM policy refined for Pipeline Loader
  - Without ControlNet: keep Text Encoder (TE) and VAE resident on GPU; only offload transformer (DiT)
  - With ControlNet: manage TE/VAE as before (offload pipe includes transformer+TE+VAE) to avoid 99% VRAM stalls
- Example workflow added: 09_qwen_controlnet_inpaint.json

## Installation
1) Clone into ComfyUI/custom_nodes
   - git clone https://github.com/your-repo/ComfyUI_Qwen-Image.git
2) Install dependencies
   - pip install -r requirements.txt
3) Install DiffSynth-Studio (required for DiffSynth nodes)
   - git clone https://github.com/modelscope/DiffSynth-Studio.git
   - cd DiffSynth-Studio && pip install -e .
4) (Optional) MMGP offload (if available on your platform). If not present, the plugin falls back to built‑in VRAM management.
5) Restart ComfyUI

Notes
- First run will auto-download missing model weights via ModelScope/model ids used by DiffSynth.
- A CUDA-capable GPU is recommended; bf16 is the default precision.

## Nodes (DiffSynth section)
- QwenImageDiffSynthLoRALoader: Load one LoRA
- QwenImageDiffSynthLoRAMulti: Merge two LoRA inputs
- QwenImageDiffSynthControlNetLoader: Load Blockwise ControlNet
  - Types: canny, depth, pose, normal, seg, inpaint
  - You can choose a local .safetensors or just pick a type to use the official repo id
- QwenImageDiffSynthPipelineLoader: Main pipeline with VRAM optimization and LoRA/ControlNet integration
  - base_model: auto | Qwen-Image | Qwen-Image-EliGen | Qwen-Image-Distill-Full
  - torch_dtype: bfloat16 | float16 | float32 (bf16 recommended)
  - offload_to_cpu: true/false
  - vram_optimization: No_Optimization | HighRAM_HighVRAM | HighRAM_LowVRAM | LowRAM_HighVRAM | LowRAM_LowVRAM | VerylowRAM_LowVRAM
- QwenDiffSynthSetControlArgs: Create control_args from reference image (+ optional inpaint_mask)
- QwenImageDiffSynthControlNetInput: One-stop preprocessing for control images (canny/depth/pose/normal/seg)
- QwenImageDiffSynthAdvancedSampler: Advanced sampler with EliGen + Blockwise ControlNet support
- QwenImageDiffSynthMemoryManager / pass-through helpers

## ControlNet Inpaint Usage
There are two recommended ways:

A) Minimal wiring (recommended)
- Use QwenDiffSynthSetControlArgs
  - ref_image: the reference image to inpaint around
  - inpaint_mask: white=areas to modify, black=areas to preserve
- In QwenImageDiffSynthControlNetLoader, set controlnet_type to inpaint
  - If you don’t provide a local file, the loader will add the official model id DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint
- Connect control_args to Advanced Sampler

B) Custom preprocessing path
- Build your own mask image and pass it through QwenDiffSynthSetControlArgs, then proceed as above.

Advanced Sampler will construct ControlNetInput(image=..., inpaint_mask=...) and handle size/mode normalization for the mask.

## Sampler Behavior (Quality/Speed)
- Distill models
  - cfg_scale fixed ~1.0 recommended internally (handled automatically)
  - Steps: respect user setting; capped at 15 if a higher value is set; enhanced_quality does NOT force higher steps
- Lightning LoRA (detected by filename containing "lightning")
  - Respect user steps (e.g., 4/8). enhanced_quality only slightly increases CFG
- Original models
  - enhanced_quality may raise steps to a minimum of 25 and boost CFG; fast reduces both moderately

## VRAM Optimization Policy
- Without ControlNet
  - TE + VAE stay on GPU; only transformer is offloaded/profiled
  - MMGP budgets around transformer=90%, workingVRAM≈70%
- With ControlNet
  - TE + VAE managed as original (included in offload/profile set when offload_to_cpu true)
  - MMGP budgets default back to transformer/text_encoder/vae and a more conservative workingVRAM≈25%
- If MMGP isn’t available, we automatically fall back to DiffSynth’s built‑in enable_vram_management()

## Example Workflows
- example_workflows/09_qwen_controlnet_inpaint.json — Inpaint ControlNet end‑to‑end
- example_workflows/qwen_diffsynth_controlnet_workflow.json — Multi‑ControlNet and Distill pipeline example
- example_workflows/03_qwen_controlnet_canny.json — Canny control
- example_workflows/04_qwen_controlnet_depth.json — Depth control
- example_workflows/05_qwen_eligen_entities.json — EliGen entity prompts + masks

## Tips
- If VRAM hits 99% and slows down while using ControlNet, keep offload_to_cpu=true and use the provided profiles; avoid pinning TE/VAE to GPU when ControlNet is active
- Inpaint masks: use binary masks where possible; white=edit area, black=keep
- For higher speed on Distill, consider 8–12 steps; for original models, enhanced_quality at ≥25 steps for best quality

## Troubleshooting
- “Pipeline has no Blockwise ControlNet loaded; ignoring control_args”
  - Ensure you selected a ControlNet model or set controlnet_type in the loader
- Mask alignment issues
  - The sampler will auto-resize the mask to the control image size; provide reasonably aligned inputs for best results
- Extremely low VRAM usage without ControlNet
  - This is expected if offloading is enabled; raise working size (resolution) or steps slightly, or disable offloading if you have headroom

## License
Apache 2.0. See LICENSE.

## Acknowledgements
- DiffSynth-Studio by ModelScope
- Community repos and prior workflows that inspired this integration

# ComfyUI Qwen-Image Plugin v2.1

🎨 A comprehensive ComfyUI plugin for Qwen-Image model integration using ComfyUI's standard separated model loading architecture. Features exceptional Chinese text rendering, advanced image generation capabilities, and the new **Advanced Diffusion Loader** with comprehensive optimization options.

## 🌟 Features

### ⚡ New in v2.1: Advanced Features
- **🚀 All-in-One Loading**: Integrated UNet, CLIP, and VAE loading in a single node
- **🎯 Performance Optimization**: Advanced weight and compute data type selection
- **🧠 SageAttention Support**: Memory-efficient attention mechanisms
- **⚙️ cuBLAS Integration**: Hardware-accelerated linear layer computations
- **💾 Memory Management**: FP16 accumulation and auto-detection features
- **🔧 Advanced Configuration**: Extra state dictionary support for power users
- **🎭 Dual CLIP Support**: Load and use two CLIP models simultaneously for enhanced text understanding
- **🔀 Flexible Blending**: Multiple modes to combine dual CLIP outputs (average, concat, etc.)

### New Architecture (v2.0+)
- **Separated Model Loading**: Uses ComfyUI's standard UNet/CLIP/VAE loading system
- **Standard Workflow Integration**: Fully compatible with ComfyUI's native sampling and conditioning
- **Optimized Performance**: Better memory management and faster loading times
- **Model Flexibility**: Support for different precision formats (bf16, fp8, fp8_e4m3fn, fp8_e5m2)

### Core Capabilities
- **Advanced Image Generation**: High-quality text-to-image generation with Qwen-Image model
- **Exceptional Chinese Text Rendering**: Industry-leading Chinese character and text rendering in images
- **🎯 DiffSynth Integration**: ControlNet support, LoRA integration, and advanced memory management
- **ControlNet Support**: Canny, Depth, Pose, Normal, and Segmentation control
- **LoRA Integration**: Load and combine multiple LoRA models with weight control
- **Lightning LoRA**: Fast inference optimization for improved performance
- **Intelligent Image Editing**: Style transfer, object manipulation, and detail enhancement (legacy)
- **Multi-modal Understanding**: Image analysis, object detection, and content comprehension (legacy)
- **Professional Text Rendering**: High-quality text overlay with customizable fonts and styles (legacy)

### Key Advantages
- **Chinese Language Optimization**: Specifically optimized for Chinese text processing and rendering
- **Multiple Aspect Ratios**: Support for various aspect ratios optimized for different use cases
- **Flexible Configuration**: Extensive customization options for generation parameters
- **ComfyUI Integration**: Seamless integration with ComfyUI's node-based workflow system
- **High Performance**: Optimized for both quality and speed

## 📦 Installation

### Prerequisites
- ComfyUI installed and running
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB VRAM for optimal performance

### Required Models
Place these models in the corresponding ComfyUI directories:

**Diffusion Models** (`ComfyUI/models/diffusion_models/`):
- `qwen_image_bf16.safetensors`
- `qwen_image_fp8_e4m3fn.safetensors`

**Text Encoders** (`ComfyUI/models/text_encoders/`):
- `qwen_2.5_vl_7b.safetensors`
- `qwen_2.5_vl_7b_fp8_scaled.safetensors`

**VAE** (`ComfyUI/models/vae/`):
- `qwen_image_vae.safetensors`

**ControlNet Models** (`ComfyUI/models/controlnet/`) - For DiffSynth features:
- `qwen_image_blockwise_controlnet_canny.safetensors`
- `qwen_image_blockwise_controlnet_depth.safetensors`

**LoRA Models** (`ComfyUI/models/loras/`) - For DiffSynth features:
- `qwen_image_distill.safetensors`
- `qwen_image_lightning.safetensors`

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "Qwen-Image"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation
1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/ComfyUI_Qwen-Image.git
   ```

3. Install dependencies:
   ```bash
   cd ComfyUI_Qwen-Image
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## 🚀 Quick Start

### Advanced Workflow (v2.1 - Recommended)
1. Add **🎨 Qwen-Image Advanced Diffusion Loader** node
2. Configure optimization settings (weight dtype, SageAttention, etc.)
3. Add **🎨 Qwen-Image Text Encode** nodes for positive and negative prompts
4. Add **🎨 Qwen-Image Empty Latent** node
5. Add **🎨 Qwen-Image Sampler** node
6. Add **VAE Decode** and **Save Image** nodes
7. Connect the workflow and execute

### Standard Workflow (Individual Loaders)
1. Add **🎨 Qwen-Image UNet Loader** node
2. Add **🎨 Qwen-Image CLIP Loader** node
3. Add **🎨 Qwen-Image VAE Loader** node
4. Add **🎨 Qwen-Image Text Encode** nodes for positive and negative prompts
5. Add **🎨 Qwen-Image Empty Latent** node
6. Add **🎨 Qwen-Image Sampler** node
7. Add **VAE Decode** and **Save Image** nodes
8. Connect the workflow and execute

### Example Workflow Files
- `qwen_diffsynth_controlnet_workflow.json` - 使用 DiffSynth 管线，已支持 base_model（auto/Qwen-Image/Qwen-Image-EliGen/Qwen-Image-Distill-Full）
- `05_qwen_eligen_entities.json` - EliGen 实体控制示例（将 base_model 切到 Qwen-Image-EliGen）
- `dual_clip_workflow.json` - Dual CLIP text encoding with enhanced understanding
- `advanced_diffusion_loader_workflow.json` - Advanced loader with optimizations
- `qwen_image_standard_workflow.json` - Basic text-to-image generation
- `chinese_text_rendering_workflow.json` - Optimized for Chinese calligraphy

### Example Prompt (Chinese)
```
一只可爱的小猫咪坐在樱花树下，春天的阳光洒在它的毛发上，背景是传统的中式庭院
```

### Example Prompt (English)
```
A beautiful landscape with Chinese text '你好世界' written in elegant calligraphy
```

## 📚 Node Reference

### Model Loaders

#### 🎨 Qwen-Image Advanced Diffusion Loader ⭐ NEW
**All-in-one model loader with advanced optimization options.**
- **Inputs**:
  - Model name, weight dtype, compute dtype
  - SageAttention mode, cuBLAS modifications
  - Auto-detection for CLIP and VAE
  - Extra state dictionary for advanced configs
- **Outputs**: MODEL, CLIP, VAE
- **Features**:
  - Memory optimization for various GPU tiers
  - Performance tuning options
  - Auto-detection of compatible models
- **See**: `ADVANCED_DIFFUSION_LOADER_GUIDE.md` for detailed usage

#### 🎨 Qwen-Image UNet Loader
Loads the Qwen-Image diffusion model.
- **Input**: UNet model file (qwen_image_bf16.safetensors, etc.)
- **Output**: MODEL
- **Weight Types**: default, fp8_e4m3fn, fp8_e4m3fn_fast, fp8_e5m2

#### 🎨 Qwen-Image CLIP Loader ⭐ Enhanced
Loads Qwen text encoder model(s) with dual CLIP support.
- **Inputs**:
  - Primary CLIP model file
  - `load_dual_clip`: Enable dual CLIP loading
  - Secondary CLIP model file (optional)
  - Device selection
- **Outputs**: Primary CLIP, Secondary CLIP
- **Features**: Single or dual CLIP loading based on user choice

#### 🎨 Qwen-Image VAE Loader
Loads the Qwen VAE model.
- **Input**: VAE model file (qwen_image_vae.safetensors)
- **Output**: VAE

### Text Processing

#### 🎨 Qwen-Image Text Encode
Encodes text prompts with Chinese optimization.
- **Inputs**: CLIP, text
- **Output**: CONDITIONING
- **Features**: Magic prompt enhancement, language detection

#### 🎨 Qwen-Image Dual Text Encode ⭐ NEW
Advanced text encoder supporting dual CLIP models.
- **Inputs**:
  - Primary CLIP, text prompt
  - Secondary CLIP (optional)
  - Blend mode selection
  - Magic prompt and language options
- **Output**: CONDITIONING
- **Features**:
  - Dual CLIP text encoding
  - Multiple blend modes (average, concat, primary_only, secondary_only)
  - Enhanced text understanding
- **See**: `DUAL_CLIP_GUIDE.md` for detailed usage

### Latent Operations

#### 🎨 Qwen-Image Empty Latent
Creates empty latent images with optimized dimensions.
- **Inputs**: width, height, aspect_ratio, batch_size
- **Output**: LATENT
- **Aspect Ratios**: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3

#### 🎨 Qwen-Image Sampler
Samples images using ComfyUI's standard sampling system.
- **Inputs**: MODEL, positive, negative, latent_image
- **Output**: LATENT
- **Parameters**: seed, steps, cfg, sampler_name, scheduler, denoise

## 🎯 Optimized Aspect Ratios

| Ratio | Dimensions | Use Case |
|-------|------------|----------|
| 1:1   | 1328×1328  | Square images, social media |
| 16:9  | 1664×928   | Landscape, wallpapers |
| 9:16  | 928×1664   | Portrait, mobile screens |
| 4:3   | 1472×1104  | Traditional photos |
| 3:4   | 1104×1472  | Portrait photos |
| 3:2   | 1584×1056  | Photography standard |
| 2:3   | 1056×1584  | Book covers, posters |

## 🔧 Advanced Configuration

### Magic Prompts
The plugin automatically enhances prompts with quality improvements:
- **Chinese**: "超清，4K，电影级构图"
- **English**: "Ultra HD, 4K, cinematic composition."

### Language Detection
Automatic language detection based on Unicode character ranges:
- Chinese characters (U+4E00-U+9FFF) trigger Chinese optimizations
- Other characters use English optimizations

## 🆕 Migration from v1.0

### Breaking Changes
- Old `QwenImageModelLoader` and `QwenImageGenerate` nodes are deprecated
- New separated loading architecture required
- Workflow files need to be updated

### Legacy Support
Legacy nodes are still available for backward compatibility:
- `QwenImageEdit` (Legacy)
- `QwenImageTextRender` (Legacy)
- `QwenImageUnderstand` (Legacy)

## 🐛 Troubleshooting

### Common Issues
1. **Model not found**: Ensure models are in correct ComfyUI directories
2. **CUDA out of memory**: Use fp8 models or reduce batch size
3. **Text encoding errors**: Check CLIP model is loaded correctly

### Performance Tips
- Use fp8 models for lower VRAM usage
- Enable magic prompts for better quality
- Use appropriate aspect ratios for your use case
- For DiffSynth features: Choose appropriate VRAM optimization strategy
- Use Lightning LoRA for faster inference when quality is acceptable

## 🎯 Advanced Features

### DiffSynth Integration
For advanced ControlNet and LoRA features, see the [DiffSynth Guide](QWEN_DIFFSYNTH_GUIDE.md).

**Key DiffSynth Features:**
- **ControlNet Support**: Structure control with Canny, Depth, Pose, Normal, Segmentation
- **LoRA Integration**: Load and combine multiple LoRA models
- **Memory Management**: Advanced VRAM optimization strategies
- **Lightning LoRA**: Fast inference optimization

**DiffSynth Nodes:**
- `QwenImageDiffSynthLoRALoader`: Load LoRA models
- `QwenImageDiffSynthControlNetLoader`: Load ControlNet models
- `QwenImageDiffSynthPipelineLoader`: Main pipeline with memory management（新增 base_model：auto / Qwen-Image / Qwen-Image-EliGen / Qwen-Image-Distill-Full）
- `QwenImageDiffSynthDistillPipelineLoader`: 专用 Distill-Full 管线加载器（保持不变）
- `QwenImageDiffSynthSampler`: Generate images with ControlNet/LoRA
- `QwenImageDiffSynthMemoryManager`: Advanced memory optimization

## 📄 License

Apache 2.0 License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## 📞 Support

- GitHub Issues: Report bugs and feature requests
- Documentation: Check example workflows
- Community: Join ComfyUI Discord for discussions
