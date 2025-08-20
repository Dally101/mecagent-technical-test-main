import gradio as gr
import torch
from transformers import pipeline
from PIL import Image
import time
import traceback

# Mock spaces module for local testing
class MockSpaces:
    @staticmethod
    def GPU(duration=60):
        def decorator(func):
            return func
        return decorator

spaces = MockSpaces()

# Global model storage
models = {}

def load_glm_model(model_choice):
    """Load GLM model on GPU."""
    model_map = {
        "GLM-4.5V-AWQ": "QuantTrio/GLM-4.5V-AWQ",
        "GLM-4.5V-FP8": "zai-org/GLM-4.5V-FP8",
        "GLM-4.5V": "zai-org/GLM-4.5V"
    }
    
    model_name = model_map[model_choice]
    
    if model_name in models:
        return True, f"✅ {model_choice} already loaded"
    
    try:
        # Mock for local testing - just return success
        return True, f"✅ {model_choice} loaded successfully (mock)"
        
    except Exception as e:
        error_msg = f"❌ Failed to load {model_choice}: {str(e)[:200]}"
        return False, error_msg

@spaces.GPU(duration=120)
def generate_cadquery_code(image, model_choice, prompt_style):
    """Generate CADQuery code from image."""
    
    if image is None:
        return "❌ Please upload an image first."
    
    # Mock generation for local testing
    mock_code = """import cadquery as cq

# Create a simple box
result = (cq.Workplane("XY")
          .box(10, 10, 5)
          .edges("|Z")
          .fillet(1))"""
    
    output = f"""## 🎯 Generated CADQuery Code

```python
{mock_code}
```

## 📊 Generation Info
- **Model**: {model_choice} (mock)
- **Time**: 2.5 seconds (mock)
- **Prompt**: {prompt_style}
- **Device**: {"GPU" if torch.cuda.is_available() else "CPU"}

## 🔧 Usage
```bash
pip install cadquery
python your_script.py
```

## ⚠️ Note
This is a mock response for local testing.
"""
    
    return output

def extract_cadquery_code(generated_text: str) -> str:
    """Extract clean CADQuery code from generated text."""
    text = generated_text.strip()
    
    if "```python" in text:
        start = text.find("```python") + 9
        end = text.find("```", start)
        if end > start:
            code = text[start:end].strip()
        else:
            code = text[start:].strip()
    elif "import cadquery" in text.lower():
        lines = text.split('\n')
        code_lines = []
        started = False
        
        for line in lines:
            if "import cadquery" in line.lower():
                started = True
            if started:
                code_lines.append(line)
        
        code = '\n'.join(code_lines)
    else:
        code = text
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('```'):
            cleaned_lines.append(line)
    
    final_code = '\n'.join(cleaned_lines)
    
    if "import cadquery" not in final_code:
        final_code = "import cadquery as cq\n\n" + final_code
    
    if "result" not in final_code and "=" in final_code:
        lines = final_code.split('\n')
        for i, line in enumerate(lines):
            if "=" in line and ("cq." in line or "Workplane" in line):
                lines[i] = f"result = {line.split('=', 1)[1].strip()}"
                break
        final_code = '\n'.join(lines)
    
    return final_code

def test_model_loading(model_choice):
    """Test loading a specific model."""
    success, message = load_glm_model(model_choice)
    return f"## Test Result\n\n{message}"

def get_system_info():
    """Get system information."""
    info = {
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Device Count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "PyTorch Version": torch.__version__,
        "Device": "GPU" if torch.cuda.is_available() else "CPU"
    }
    
    info_text = "## 🖥️ System Information\n\n"
    for key, value in info.items():
        info_text += f"- **{key}**: {value}\n"
    
    return info_text

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="GLM-4.5V CAD Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔧 GLM-4.5V CAD Generator (Local Test)
        
        Upload a 3D CAD model image and generate CADQuery Python code using GLM-4.5V models!
        
        **Available Models:**
        - **GLM-4.5V-AWQ**: AWQ quantized (fastest startup)
        - **GLM-4.5V-FP8**: 8-bit quantized (balanced)
        - **GLM-4.5V**: Full precision (best quality)
        
        *Note: This is a local test version with mock responses.*
        """)
        
        with gr.Tab("🚀 Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil", 
                        label="Upload CAD Model Image",
                        height=400
                    )
                    
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
                    
                    generate_btn = gr.Button("🚀 Generate CADQuery Code", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    output_text = gr.Markdown(
                        label="Generated Code",
                        value="Upload an image and click 'Generate' to start!"
                    )
            
            generate_btn.click(
                fn=generate_cadquery_code,
                inputs=[image_input, model_choice, prompt_style],
                outputs=output_text
            )
        
        with gr.Tab("🧪 Test"):
            with gr.Row():
                with gr.Column():
                    test_model_choice = gr.Dropdown(
                        choices=["GLM-4.5V-AWQ", "GLM-4.5V-FP8", "GLM-4.5V"],
                        value="GLM-4.5V-AWQ",
                        label="Model to Test"
                    )
                    test_btn = gr.Button("🧪 Test Model Loading", variant="secondary")
                
                with gr.Column():
                    test_output = gr.Markdown(value="Click 'Test Model Loading' to check if models work.")
            
            test_btn.click(
                fn=test_model_loading,
                inputs=test_model_choice,
                outputs=test_output
            )
        
        with gr.Tab("⚙️ System"):
            info_output = gr.Markdown()
            refresh_btn = gr.Button("🔄 Refresh System Info")
            
            demo.load(fn=get_system_info, outputs=info_output)
            refresh_btn.click(fn=get_system_info, outputs=info_output)
        
        with gr.Tab("📖 Help"):
            gr.Markdown("""
            ## 🎯 How to Use
            
            1. **Upload Image**: Clear 3D CAD model images work best
            2. **Select Model**: GLM-4.5V-AWQ is fastest for testing
            3. **Choose Prompt**: Chain-of-Thought usually gives best results
            4. **Generate**: Click the button and wait for results
            
            ## 💡 Tips for Best Results
            
            - Use clear, well-lit CAD images
            - Simple geometric shapes work better than complex assemblies
            - Try different prompt styles if first attempt isn't satisfactory
            
            ## 🔧 Using Generated Code
            
            ```bash
            # Install CADQuery
            pip install cadquery
            
            # Run your generated code
            python your_cad_script.py
            
            # Export to STL
            cq.exporters.export(result, "model.stl")
            ```
            
            ## 🖥️ Hardware Requirements
            
            - This app runs on GPU-enabled Hugging Face Spaces
            - First model load takes 5-10 minutes
            - Generation takes 15-45 seconds per image
            """)
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting GLM-4.5V CAD Generator (Local Test)...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )