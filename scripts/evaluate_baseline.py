"""
Baseline Model Evaluation Script

Runs a base LLM (Mistral-7B or Llama-3-8B) on the test set and calculates BLEU score.
This establishes the baseline before fine-tuning.

Usage:
    python scripts/evaluate_baseline.py --model mistral --samples 100
    python scripts/evaluate_baseline.py --model llama --samples 2000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_test_set(path: Path, num_samples: int = None) -> list[dict]:
    """Load test set from JSONL file."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            if num_samples and len(samples) >= num_samples:
                break
    return samples


def get_model_and_tokenizer(model_name: str):
    """Load the specified model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_ids = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    
    model_id = model_ids.get(model_name.lower())
    if not model_id:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_ids.keys())}")
    
    print(f"Loading model: {model_id}")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load with appropriate settings
    if device == "cuda":
        # Use 4-bit quantization for memory efficiency
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("WARNING: Running on CPU will be very slow!")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def generate_translation(model, tokenizer, instruction: str, input_text: str, device: str) -> str:
    """Generate translation using the model."""
    import torch
    
    # Format as chat/instruction
    if "mistral" in model.config._name_or_path.lower():
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
    else:  # Llama format
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def calculate_bleu(predictions: list[str], references: list[str]) -> dict:
    """Calculate BLEU score using sacrebleu."""
    import sacrebleu
    
    # sacrebleu expects references as list of lists
    refs = [[ref] for ref in references]
    
    bleu = sacrebleu.corpus_bleu(predictions, [[r[0] for r in refs]])
    
    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
        "precisions": bleu.precisions,
        "brevity_penalty": bleu.bp,
        "length_ratio": bleu.sys_len / bleu.ref_len if bleu.ref_len > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on test set")
    parser.add_argument("--model", type=str, default="mistral", choices=["mistral", "llama"],
                        help="Base model to evaluate")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of test samples to evaluate (default: 100)")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to test JSONL file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results")
    args = parser.parse_args()
    
    # Paths
    test_file = Path(args.test_file) if args.test_file else PROJECT_ROOT / "data" / "test_set.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Baseline Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Test file: {test_file}")
    print(f"Samples: {args.samples}")
    print()
    
    # Load test set
    print("Loading test set...")
    test_samples = load_test_set(test_file, args.samples)
    print(f"Loaded {len(test_samples)} samples")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, device = get_model_and_tokenizer(args.model)
    print("Model loaded!")
    
    # Run inference
    print("\nRunning inference...")
    predictions = []
    references = []
    
    for sample in tqdm(test_samples, desc="Generating translations"):
        pred = generate_translation(
            model, tokenizer,
            sample["instruction"],
            sample["input"],
            device
        )
        predictions.append(pred)
        references.append(sample["output"])
    
    # Calculate BLEU
    print("\nCalculating BLEU score...")
    bleu_results = calculate_bleu(predictions, references)
    
    # Prepare results
    results = {
        "model": args.model,
        "model_id": model.config._name_or_path,
        "num_samples": len(test_samples),
        "timestamp": datetime.now().isoformat(),
        "bleu_score": bleu_results["bleu"],
        "bleu_details": bleu_results,
        "sample_predictions": [
            {
                "input": test_samples[i]["input"][:100] + "...",
                "reference": references[i][:100] + "...",
                "prediction": predictions[i][:100] + "..."
            }
            for i in range(min(5, len(predictions)))
        ]
    }
    
    # Save results
    output_file = output_dir / f"baseline_bleu_{args.model}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"BLEU Score: {bleu_results['bleu']:.2f}")
    print(f"Precisions: {[f'{p:.1f}' for p in bleu_results['precisions']]}")
    print(f"Brevity Penalty: {bleu_results['brevity_penalty']:.3f}")
    print(f"\nResults saved to: {output_file}")
    
    # Print sample predictions
    print("\nSample Predictions:")
    for i in range(min(3, len(predictions))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {test_samples[i]['input'][:80]}...")
        print(f"Reference: {references[i][:80]}...")
        print(f"Prediction: {predictions[i][:80]}...")
    
    return results


if __name__ == "__main__":
    main()
