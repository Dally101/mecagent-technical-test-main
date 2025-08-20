"""
GLM-4.5V CAD Generation Fine-tuning Pipeline
Inspired by HuggingFace TRL examples and best practices for VLM training
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, TaskType
import gradio as gr
from PIL import Image
import json
from typing import Dict, List, Optional

# Configuration
MODEL_ID = "zai-org/GLM-4.5V"  # or "QuantTrio/GLM-4.5V-AWQ" for faster training
DATASET_NAME = "CADCODER/GenCAD-Code"  # Our CAD dataset
OUTPUT_DIR = "./glm-cad-finetuned"

class CADVisionDataCollator:
    """Custom data collator for vision-language CAD generation."""
    
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
    
    def __call__(self, examples):
        # Process images and text together
        texts = []
        images = []
        
        for example in examples:
            # Format conversation for CAD generation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Generate CADQuery Python code for this 3D model:"}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": example["code"]}
                    ]
                }
            ]
            
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False))
            images.append(example["image"])
        
        # Process batch
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Add labels for training
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100
        
        return batch

def prepare_dataset():
    """Load and prepare the CAD dataset for training."""
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train[:1000]")  # Start with 1000 samples
    
    def preprocess_function(examples):
        """Preprocess CAD examples."""
        processed = {
            "image": [],
            "code": [],
            "conversation": []
        }
        
        for i in range(len(examples["image"])):
            # Ensure image is PIL
            img = examples["image"][i]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            
            processed["image"].append(img)
            processed["code"].append(examples["code"][i])
            
            # Create conversation format
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Generate CADQuery Python code for this 3D model:"
                    },
                    {
                        "role": "assistant",
                        "content": examples["code"][i]
                    }
                ]
            }
            processed["conversation"].append(json.dumps(conversation))
        
        return processed
    
    # Process dataset
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return dataset

def setup_model_for_training():
    """Setup GLM model with LoRA for efficient fine-tuning."""
    
    # Quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # LoRA configuration for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor

def train_model(model, processor, dataset):
    """Train the model using SFTTrainer."""
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=25,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        push_to_hub=False,  # Set True to push to HF Hub
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False
    )
    
    # Create data collator
    data_collator = CADVisionDataCollator(processor, model)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        peft_config=lora_config if not hasattr(model, 'peft_config') else None,
        dataset_text_field="conversation",  # Field containing the conversations
        max_seq_length=2048,
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    return trainer

def create_gradio_interface(model_path=OUTPUT_DIR):
    """Create a Gradio interface for the fine-tuned model."""
    
    # Load fine-tuned model
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    def generate_cad_code(image, temperature=0.7, max_length=512):
        """Generate CAD code from image."""
        
        # Prepare input
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Generate CADQuery Python code for this 3D model:"}
            ]
        }]
        
        # Process
        inputs = processor(
            text=processor.apply_chat_template(messages, tokenize=False),
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
        
        # Decode
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code
        if "```python" in generated_text:
            start = generated_text.find("```python") + 9
            end = generated_text.find("```", start)
            code = generated_text[start:end] if end > start else generated_text[start:]
        else:
            code = generated_text
        
        return code
    
    # Create interface
    interface = gr.Interface(
        fn=generate_cad_code,
        inputs=[
            gr.Image(type="pil", label="CAD Model Image"),
            gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
            gr.Slider(128, 1024, value=512, label="Max Length")
        ],
        outputs=gr.Code(language="python", label="Generated CADQuery Code"),
        title="Fine-tuned GLM-4.5V CAD Generator",
        description="Generate CADQuery code from 3D model images using fine-tuned GLM-4.5V"
    )
    
    return interface

def main():
    """Main training pipeline."""
    
    print("🚀 Starting GLM-4.5V CAD Fine-tuning Pipeline")
    
    # Step 1: Prepare dataset
    print("📊 Preparing dataset...")
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)} samples")
    
    # Step 2: Setup model
    print("🔧 Setting up model with LoRA...")
    model, processor = setup_model_for_training()
    
    # Step 3: Train
    print("🏋️ Starting training...")
    trainer = train_model(model, processor, dataset)
    
    # Step 4: Evaluate
    print("📈 Training complete! Results:")
    print(f"Final loss: {trainer.state.log_history[-1]['loss']:.4f}")
    
    # Step 5: Create interface
    print("🎨 Creating Gradio interface...")
    interface = create_gradio_interface()
    
    print("✅ Fine-tuning complete! Model saved to:", OUTPUT_DIR)
    print("Launch interface with: interface.launch()")
    
    return trainer, interface

if __name__ == "__main__":
    # For HuggingFace Spaces deployment, use:
    # trainer, interface = main()
    # interface.launch(share=True)
    
    print("""
    📝 To run this training pipeline:
    
    1. Install requirements:
       pip install transformers trl peft datasets accelerate bitsandbytes gradio
    
    2. Run training:
       python train_glm_cad.py
    
    3. For multi-GPU:
       accelerate launch train_glm_cad.py
    
    4. Monitor with TensorBoard:
       tensorboard --logdir ./glm-cad-finetuned
    """)