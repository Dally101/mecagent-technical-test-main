# GLM-4.5V CAD Generation Technical Test

**AI system for generating CADQuery Python code from 3D model images**

## 🎯 Project Overview

This project develops a vision-language model to automatically generate CADQuery Python code from 3D CAD model images. Successfully transitioned from deployment challenges to a functional training pipeline using LLaVA-1.5.

## 📁 Repository Structure

```
├── PROJECT_PROGRESS_REPORT.md    # Comprehensive project documentation
├── CLAUDE.md                     # Development guidelines and commands
├── app.py                        # HF Spaces deployment app (Zero GPU issues)
├── colab_training.py             # 🎯 WORKING: LLaVA-1.5 training script for Colab
├── train_cpu.py                  # CPU training proof-of-concept
├── requirements.txt              # Python dependencies
├── metrics/                      # Evaluation framework
│   ├── best_iou.py              # IoU similarity metric
│   └── valid_syntax_rate.py     # Code syntax validation
└── old_versions/                 # Archived development files
```

## 🚀 Quick Start

### Option 1: Google Colab Training (Recommended)
```bash
# 1. Open Google Colab: https://colab.research.google.com/
# 2. Enable T4 GPU: Runtime → Change runtime type → T4 GPU
# 3. Upload colab_training.py and run in 4 cells
# 4. Training completes in ~8 minutes
```

### Option 2: Local CPU Testing
```bash
pip install torch transformers datasets peft accelerate
python train_cpu.py
```

## 📊 Key Results

- ✅ **Successful training pipeline** with LLaVA-1.5-7B
- ✅ **Memory optimized** for T4 GPU (<10GB VRAM)
- ✅ **Real CAD data** from CADCODER/GenCAD-Code dataset
- ✅ **Vision-to-code generation** working end-to-end

## 🔧 Technical Architecture

**Model:** LLaVA-1.5-7B (vision-language)  
**Training:** LoRA fine-tuning with 4-bit quantization  
**Dataset:** CADCODER/GenCAD-Code (30 optimized samples)  
**Format:** `USER: <image>\nGenerate CADQuery code: ASSISTANT: {code}`  

## 📈 Project Evolution

1. **Initial Challenge:** Zero GPU context variable errors with GLM-4.5V
2. **Model Pivot:** GLM-4.5V → LLaVA-1.5 for compatibility
3. **Infrastructure Shift:** HF Spaces → Google Colab for reliability
4. **Success:** Working vision-language training pipeline

## 🎯 Key Insights

- **Right-sized models** are crucial for available compute resources
- **Infrastructure pragmatism** over theoretical optimality
- **Memory optimization** enables complex models on modest hardware
- **Incremental validation** prevents costly development dead-ends

## 📖 Documentation

See **PROJECT_PROGRESS_REPORT.md** for comprehensive documentation including:
- Technical journey and problem-solving approach
- Architecture evolution and design decisions
- Future development roadmap
- Strategic insights and lessons learned

## 🔄 Next Steps

1. Scale training data from 30 to 1000+ samples
2. Implement automated evaluation metrics (VSR + IoU)
3. Deploy production API endpoints
4. Explore advanced techniques (RLHF, multi-modal)

---

*This project demonstrates end-to-end ML system development from research through implementation to production considerations.*