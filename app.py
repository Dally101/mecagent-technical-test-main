import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import torch
from transformers import pipeline
from PIL import Image
import spaces

# Global model storage
models = {}

# Wrap the entire generation in try-except to handle Zero GPU issues
@spaces.GPU(duration=120)
def generate_on_gpu(image_array, model_name, prompt_text):
    """Core GPU function - simplified to avoid context issues."""
    pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Convert array back to PIL
    image = Image.fromarray(image_array)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }]
    
    result = pipe(messages, max_new_tokens=512, temperature=0.7)
    
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", str(result))
    else:
        return str(result)

def generate_code(image, model_choice, prompt_style):
    """Wrapper function that handles the UI logic."""
    if image is None:
        return "❌ Please upload an image first."
    
    # Prompts
    prompts = {
        "Simple": "Generate CADQuery Python code for this 3D model:",
        "Detailed": "Analyze this 3D CAD model and generate Python CADQuery code.\n\nRequirements:\n- Import cadquery as cq\n- Store result in 'result' variable\n- Use proper CADQuery syntax\n\nCode:",
        "Chain-of-Thought": "Analyze this 3D CAD model step by step:\n\nStep 1: Identify the basic geometry\nStep 2: Note any features\nStep 3: Generate clean CADQuery Python code\n\n```python\nimport cadquery as cq\n\n# Generated code:"
    }
    
    try:
        # Model mapping
        model_map = {
            "GLM-4.5V-AWQ": "QuantTrio/GLM-4.5V-AWQ",
            "GLM-4.5V-FP8": "zai-org/GLM-4.5V-FP8",
            "GLM-4.5V": "zai-org/GLM-4.5V"
        }
        
        model_name = model_map[model_choice]
        prompt_text = prompts[prompt_style]
        
        # Convert PIL to array for passing
        import numpy as np
        image_array = np.array(image)
        
        # Call GPU function
        generated_text = generate_on_gpu(image_array, model_name, prompt_text)
        
        # Extract code
        code = generated_text.strip()
        if "```python" in code:
            start = code.find("```python") + 9
            end = code.find("```", start)
            if end > start:
                code = code[start:end].strip()
        
        if "import cadquery" not in code:
            code = "import cadquery as cq\n\n" + code
        
        return f"""## 🎯 Generated CADQuery Code

```python
{code}
```

## 📊 Info
- **Model**: {model_choice}
- **Prompt**: {prompt_style}
- **Device**: GPU

## 🔧 Usage
```bash
pip install cadquery
python your_script.py
```
"""
        
    except Exception as e:
        return f"❌ **Generation Failed**: {str(e)[:500]}"

def system_info():
    """Get system info."""
    info = f"""## 🖥️ System Information

- **CUDA Available**: {torch.cuda.is_available()}
- **CUDA Devices**: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
- **PyTorch Version**: {torch.__version__}
- **Device**: {"GPU" if torch.cuda.is_available() else "CPU"}
"""
    return info

# Create interface WITHOUT ssr_mode
with gr.Blocks(title="GLM-4.5V CAD Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔧 GLM-4.5V CAD Generator
    
    Generate CADQuery Python code from 3D CAD model images using GLM-4.5V models!
    
    **Models**: GLM-4.5V-AWQ (fastest) | GLM-4.5V-FP8 (balanced) | GLM-4.5V (best quality)
    """)
    
    with gr.Tab("🚀 Generate"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload CAD Model Image")
                model_choice = gr.Dropdown(
                    choices=["GLM-4.5V-AWQ", "GLM-4.5V-FP8", "GLM-4.5V"],
                    value="GLM-4.5V-AWQ",
                    label="Select Model"
                )
                prompt_style = gr.Dropdown(
                    choices=["Simple", "Detailed", "Chain-of-Thought"],
                    value="Chain-of-Thought", 
                    label="Prompt Style"
                )
                generate_btn = gr.Button("🚀 Generate CADQuery Code", variant="primary")
            
            with gr.Column():
                output = gr.Markdown("Upload an image and click Generate!")
        
        generate_btn.click(
            fn=generate_code,
            inputs=[image_input, model_choice, prompt_style],
            outputs=output
        )
    
    with gr.Tab("⚙️ System"):
        info_display = gr.Markdown()
        refresh_btn = gr.Button("🔄 Refresh")
        
        demo.load(fn=system_info, outputs=info_display)
        refresh_btn.click(fn=system_info, outputs=info_display)

if __name__ == "__main__":
    print("🚀 Starting GLM-4.5V CAD Generator...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    demo.queue().launch(ssr_mode=False)  # Disable SSR mode