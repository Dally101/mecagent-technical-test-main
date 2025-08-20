"""
Hugging Face Spaces Training App for GLM-4.5V CAD Generation
Simplified version for Zero GPU training with progress tracking
"""

import spaces
import gradio as gr
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import json
import os
from datetime import datetime

# Configuration
BASE_MODEL = "QuantTrio/GLM-4.5V-AWQ"  # AWQ version for faster training
DATASET_NAME = "CADCODER/GenCAD-Code"
OUTPUT_DIR = "./trained_models"

class TrainingState:
    """Global training state tracker."""
    def __init__(self):
        self.is_training = False
        self.progress = 0
        self.current_loss = 0
        self.logs = []
        self.model = None
        self.processor = None

state = TrainingState()

@spaces.GPU(duration=600)  # 10 minutes for training
def train_on_gpu(num_samples=100, epochs=1, learning_rate=5e-5):
    """Main training function that runs on GPU."""
    
    try:
        state.is_training = True
        state.logs = ["🚀 Starting training..."]
        
        # Load dataset
        state.logs.append(f"📊 Loading {num_samples} samples from dataset...")
        dataset = load_dataset(DATASET_NAME, split=f"train[:{num_samples}]")
        
        # Load model with LoRA
        state.logs.append("🔧 Loading model with LoRA configuration...")
        
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )
        
        model = AutoModelForVision2Seq.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,  # Lower rank for faster training
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q_proj", "v_proj"]  # Minimal targets
        )
        
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state.logs.append(f"✅ Model loaded with {trainable_params:,} trainable parameters")
        
        # Prepare training data
        state.logs.append("🔄 Preparing training data...")
        
        def preprocess_batch(examples):
            texts = []
            for code in examples["code"]:
                # Simple prompt template
                text = f"Generate CADQuery code:\n{code}"
                texts.append(text)
            
            # Tokenize
            model_inputs = processor.tokenizer(
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            return model_inputs
        
        # Process dataset
        tokenized_dataset = dataset.map(
            preprocess_batch,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments (minimal for demo)
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",  # No external reporting
            remove_unused_columns=False
        )
        
        # Custom callback for progress tracking
        class ProgressCallback:
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    if "loss" in logs:
                        state.current_loss = logs["loss"]
                    state.progress = (state.global_step / state.max_steps) * 100
                    state.logs.append(f"Step {state.global_step}: Loss = {logs.get('loss', 'N/A'):.4f}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=processor.tokenizer,
            callbacks=[ProgressCallback()]
        )
        
        state.logs.append("🏋️ Starting training...")
        
        # Train
        trainer.train()
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{OUTPUT_DIR}/glm_cad_{timestamp}"
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)
        
        state.model = model
        state.processor = processor
        state.logs.append(f"✅ Training complete! Model saved to {save_path}")
        state.is_training = False
        
        return "\n".join(state.logs)
        
    except Exception as e:
        state.is_training = False
        error_msg = f"❌ Training failed: {str(e)}"
        state.logs.append(error_msg)
        return "\n".join(state.logs)

@spaces.GPU(duration=60)
def generate_with_finetuned(image, prompt="Generate CADQuery code for this 3D model:"):
    """Generate code using the fine-tuned model."""
    
    if state.model is None:
        return "❌ No fine-tuned model available. Please train first!"
    
    try:
        # Prepare input
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        inputs = state.processor(
            text=state.processor.apply_chat_template(messages, tokenize=False),
            images=image,
            return_tensors="pt"
        ).to(state.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = state.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode
        generated = state.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code
        if "```python" in generated:
            start = generated.find("```python") + 9
            end = generated.find("```", start)
            code = generated[start:end] if end > start else generated
        else:
            code = generated
        
        return code
        
    except Exception as e:
        return f"❌ Generation failed: {str(e)}"

def create_training_interface():
    """Create the Gradio interface for training."""
    
    with gr.Blocks(title="GLM-4.5V CAD Training", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎓 GLM-4.5V CAD Generation Training
        
        Fine-tune GLM-4.5V on CAD generation dataset using LoRA!
        
        **Note:** Training requires GPU and will take several minutes.
        """)
        
        with gr.Tab("🏋️ Train Model"):
            with gr.Row():
                with gr.Column():
                    num_samples = gr.Slider(
                        10, 500, value=50,
                        label="Number of Training Samples",
                        info="More samples = better quality but longer training"
                    )
                    epochs = gr.Slider(
                        1, 5, value=1,
                        label="Training Epochs",
                        info="More epochs = better learning but risk of overfitting"
                    )
                    lr = gr.Number(
                        value=5e-5,
                        label="Learning Rate",
                        info="Higher = faster learning but less stable"
                    )
                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                
                with gr.Column():
                    training_output = gr.Textbox(
                        label="Training Progress",
                        lines=20,
                        value="Ready to train...",
                        interactive=False
                    )
            
            train_btn.click(
                fn=train_on_gpu,
                inputs=[num_samples, epochs, lr],
                outputs=training_output
            )
        
        with gr.Tab("🔮 Test Model"):
            gr.Markdown("Test your fine-tuned model on new images!")
            
            with gr.Row():
                with gr.Column():
                    test_image = gr.Image(type="pil", label="Upload CAD Image")
                    test_prompt = gr.Textbox(
                        value="Generate CADQuery code for this 3D model:",
                        label="Prompt"
                    )
                    generate_btn = gr.Button("Generate Code", variant="primary")
                
                with gr.Column():
                    generated_code = gr.Code(
                        language="python",
                        label="Generated CADQuery Code"
                    )
            
            generate_btn.click(
                fn=generate_with_finetuned,
                inputs=[test_image, test_prompt],
                outputs=generated_code
            )
        
        with gr.Tab("📚 Examples"):
            gr.Markdown("""
            ## Training Best Practices
            
            1. **Start Small**: Begin with 50-100 samples to test
            2. **Monitor Loss**: Loss should decrease during training
            3. **Avoid Overfitting**: Don't use too many epochs on small datasets
            4. **Test Regularly**: Generate samples to check quality
            
            ## Expected Results
            
            - **50 samples**: Basic shape recognition
            - **200 samples**: Better feature detection
            - **500+ samples**: High-quality CAD generation
            
            ## Resource Usage
            
            - Training uses Zero GPU allocation (H200)
            - Each training session limited to 10 minutes
            - Model saved locally in Space storage
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_training_interface()
    demo.launch(share=True, show_error=True)