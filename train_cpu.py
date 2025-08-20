"""
CPU-Friendly Training Script for GLM-4.5V CAD Generation - FIXED
Simplified version for testing and development
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import json
import os
from dataclasses import dataclass
from typing import Dict, List

# Simple configuration for CPU testing
CONFIG = {
    "base_model": "microsoft/DialoGPT-small",  # Small model for CPU testing
    "dataset_name": "CADCODER/GenCAD-Code",
    "output_dir": "./test-cad-model",
    "max_samples": 50,  # Very small for CPU
    "batch_size": 1,
    "gradient_accumulation": 4,
    "epochs": 1,
    "learning_rate": 5e-5,
    "max_length": 512
}

@dataclass
class SimpleDataCollator:
    """Simple data collator for text-only training."""
    tokenizer: any
    max_length: int = 512
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract texts
        texts = [f["text"] for f in features]
        
        # Tokenize
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Create labels for causal LM
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch

def prepare_simple_dataset(dataset_name: str, max_samples: int = 50):
    """Prepare a simplified text-only dataset for CPU training."""
    print(f"📊 Loading dataset: {dataset_name}")
    
    try:
        # Load small subset
        dataset = load_dataset(dataset_name, split=f"train[:{max_samples}]")
        
        # Check what columns exist
        print(f"📋 Dataset columns: {dataset.column_names}")
        
        def create_text_examples(examples):
            """Convert to text-only format."""
            texts = []
            
            # Try different possible column names
            code_column = None
            for col in ['cadquery', 'code', 'cadquery_code', 'text', 'target']:
                if col in examples:
                    code_column = col
                    break
            
            if code_column is None:
                print(f"❌ No code column found in: {list(examples.keys())}")
                raise KeyError("No code column found")
            
            print(f"✅ Using column: {code_column}")
            
            for i in range(len(examples[code_column])):
                # Create simple prompt-response format
                code = examples[code_column][i]
                text = f"Generate CADQuery code:\n{code}<|endoftext|>"
                texts.append(text)
            
            return {"text": texts}
        
        # Process dataset
        dataset = dataset.map(
            create_text_examples,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"✅ Dataset prepared: {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        
        # Create dummy dataset for testing
        print("🔄 Creating dummy dataset for testing...")
        dummy_codes = [
            "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 5)",
            "import cadquery as cq\nresult = cq.Workplane('XY').cylinder(5, 10)",
            "import cadquery as cq\nresult = cq.Workplane('XY').box(20, 15, 8).fillet(2)",
            "import cadquery as cq\nresult = cq.Workplane('XY').sphere(8)",
            "import cadquery as cq\nresult = cq.Workplane('XY').box(15, 10, 5).edges('|Z').fillet(1)",
        ]
        
        texts = [f"Generate CADQuery code:\n{code}<|endoftext|>" for code in dummy_codes]
        
        from datasets import Dataset
        # Repeat dummy data to reach max_samples
        all_texts = texts * (max_samples // len(texts) + 1)
        dataset = Dataset.from_dict({"text": all_texts[:max_samples]})
        
        print(f"✅ Dummy dataset created: {len(dataset)} samples")
        return dataset

def setup_simple_model(model_name: str):
    """Set up a simple model for CPU training."""
    print(f"🔧 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu"
    )
    
    # Simple LoRA config for CPU
    lora_config = LoraConfig(
        r=8,  # Small rank for CPU
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"]  # DialoGPT modules
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"💡 Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer

def train_simple_model(model, tokenizer, dataset, config):
    """Train the model with simple settings."""
    print("🏋️ Starting CPU training...")
    
    # Training arguments for CPU - FIXED TYPO
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        warmup_steps=10,
        logging_steps=5,
        save_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        fp16=False,  # No FP16 on CPU
        dataloader_pin_memory=False,
        # Remove use_cpu - not a valid argument
    )
    
    # Data collator
    data_collator = SimpleDataCollator(
        tokenizer=tokenizer,
        max_length=config["max_length"]
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("⏳ Training will take a few minutes on CPU...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    print(f"✅ Training complete! Model saved to {config['output_dir']}")
    return trainer

def test_simple_model(model_path: str):
    """Test the trained model."""
    print(f"🧪 Testing model: {model_path}")
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Test generation
        prompt = "Generate CADQuery code:"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("🎯 Generated:")
        print(generated)
        return generated
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return str(e)

def main():
    """Main training pipeline for CPU."""
    print("🚀 Starting CPU Training Pipeline")
    print("=" * 50)
    
    try:
        # 1. Prepare dataset
        print("\n📊 Step 1: Preparing dataset...")
        dataset = prepare_simple_dataset(CONFIG["dataset_name"], CONFIG["max_samples"])
        
        # 2. Setup model
        print("\n🔧 Step 2: Setting up model...")
        model, tokenizer = setup_simple_model(CONFIG["base_model"])
        
        # 3. Train
        print("\n🏋️ Step 3: Training...")
        trainer = train_simple_model(model, tokenizer, dataset, CONFIG)
        
        # 4. Test
        print("\n🧪 Step 4: Testing...")
        test_simple_model(CONFIG["output_dir"])
        
        print("\n🎉 Pipeline complete!")
        print(f"Model saved to: {CONFIG['output_dir']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n📝 Next steps:")
        print("1. Check the generated model in ./test-cad-model/")
        print("2. Run test_simple_model() to generate more examples")
        print("3. Once working, move to GPU version")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection for dataset download")
        print("2. Ensure you have enough disk space")
        print("3. Try reducing max_samples to 10")