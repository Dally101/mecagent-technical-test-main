"""
Google Colab Training Script - LLaVA-1.5 for CAD Generation - FIXED
Compatible with current transformers, optimized for T4 GPU
"""

# Cell 1: Setup and Installation
"""
# Run this first cell to install everything
!pip install -q transformers==4.44.1 datasets peft accelerate bitsandbytes
!pip install -q torch torchvision gradio huggingface_hub Pillow

# Enable GPU (Runtime -> Change runtime type -> T4 GPU)
import torch
import transformers
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Transformers version: {transformers.__version__}")
"""

# Cell 2: Main Training Code
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
import json
from dataclasses import dataclass
from typing import Dict, List, Any
import gradio as gr

# LLaVA-1.5 Configuration - Stable Version
CONFIG = {
    "model_name": "llava-hf/llava-1.5-7b-hf",  # Stable LLaVA model
    "dataset_name": "CADCODER/GenCAD-Code",
    "output_dir": "/content/llava-cad-trained",
    "max_samples": 30,   # Even smaller for T4
    "batch_size": 1,     
    "gradient_accumulation": 4,
    "epochs": 1,
    "learning_rate": 1e-5,  # Lower LR for stability
    "max_length": 256       # Shorter for memory
}

@dataclass
class LlavaDataCollator:
    """Data collator for LLaVA training."""
    processor: Any
    max_length: int = 256
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        
        for feature in features:
            # LLaVA format with shorter prompt
            text = f"USER: <image>\nGenerate CADQuery code: ASSISTANT: {feature['code']}"
            
            texts.append(text)
            images.append(feature["image"])
        
        # Process batch
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Labels
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100
        
        return batch

def prepare_llava_dataset(max_samples: int = 30):
    """Load and prepare small LLaVA dataset."""
    print(f"📊 Loading {max_samples} CAD examples...")
    
    dataset = load_dataset(CONFIG["dataset_name"], split=f"train[:{max_samples}]")
    print(f"📋 Dataset columns: {dataset.column_names}")
    
    def process_examples(examples):
        processed = {"image": [], "code": []}
        
        for i in range(len(examples["cadquery"])):
            # Process image
            img = examples["image"][i]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            
            # Smaller image for memory
            img = img.resize((224, 224))
            img = img.convert("RGB")
            
            # Short code snippets only
            code = examples["cadquery"][i][:200]  # Limit length
            if not code.startswith("import"):
                code = f"import cadquery as cq\n{code}"
            
            processed["image"].append(img)
            processed["code"].append(code)
        
        return processed
    
    dataset = dataset.map(
        process_examples,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ Prepared {len(dataset)} examples")
    return dataset

def setup_llava_model():
    """Setup LLaVA with aggressive quantization for T4."""
    print(f"🔧 Loading LLaVA: {CONFIG['model_name']}")
    
    # Aggressive 4-bit config for T4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(CONFIG["model_name"])
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Minimal LoRA for memory
    lora_config = LoraConfig(
        r=8,  # Smaller rank
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # Only 2 modules
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"💡 Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model, processor

def train_llava_model(model, processor, dataset):
    """Train LLaVA with minimal settings."""
    print("🏋️ Starting minimal LLaVA training...")
    
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=2,
        logging_steps=3,
        save_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        max_steps=15  # Limit steps for T4
    )
    
    data_collator = LlavaDataCollator(processor, CONFIG["max_length"])
    
    # Use older trainer parameter name
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer  # Use tokenizer instead of processing_class
    )
    
    # Train
    print("🚀 Quick training starting...")
    trainer.train()
    
    # Save
    trainer.save_model()
    processor.save_pretrained(CONFIG["output_dir"])
    
    print(f"✅ Training complete!")
    return trainer

def test_llava_model():
    """Test the trained model."""
    print("🧪 Testing LLaVA model...")
    
    try:
        processor = AutoProcessor.from_pretrained(CONFIG["output_dir"])
        model = LlavaForConditionalGeneration.from_pretrained(CONFIG["output_dir"])
        
        # Simple test
        test_image = Image.new('RGB', (224, 224), color='lightblue')
        prompt = "USER: <image>\nGenerate CADQuery code: ASSISTANT:"
        
        inputs = processor(text=prompt, images=test_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Short generation
                temperature=0.8
            )
        
        generated = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"🎯 Generated: {generated}")
        
        return generated
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return str(e)

def create_simple_interface():
    """Create simple Gradio interface."""
    def generate_code(image):
        if image is None:
            return "Upload an image"
        
        try:
            processor = AutoProcessor.from_pretrained(CONFIG["output_dir"])
            model = LlavaForConditionalGeneration.from_pretrained(CONFIG["output_dir"])
            
            prompt = "USER: <image>\nGenerate CADQuery code: ASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            
            result = processor.decode(outputs[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in result:
                code = result.split("ASSISTANT:")[-1].strip()
            else:
                code = result
            
            return code
            
        except Exception as e:
            return f"Error: {e}"
    
    interface = gr.Interface(
        fn=generate_code,
        inputs=gr.Image(type="pil", label="CAD Image"),
        outputs=gr.Textbox(label="Generated Code"),
        title="LLaVA CAD Generator"
    )
    
    return interface

def main():
    """Minimal training pipeline."""
    print("🚀 Starting Minimal LLaVA CAD Training")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ No GPU!")
        return False
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # 1. Small dataset
        dataset = prepare_llava_dataset(CONFIG["max_samples"])
        
        # 2. Model
        model, processor = setup_llava_model()
        
        # 3. Quick training
        trainer = train_llava_model(model, processor, dataset)
        
        # 4. Test
        test_llava_model()
        
        print("\n🎉 Done!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Cell 3: Run Training
if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 Next:")
        print("test_llava_model()")
        print("interface = create_simple_interface()")
        print("interface.launch(share=True)")

# Cell 4: Create Interface
"""
# Run this to create web interface
interface = create_simple_interface()
interface.launch(share=True)
"""