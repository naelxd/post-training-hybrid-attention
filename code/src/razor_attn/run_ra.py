from transformers import AutoTokenizer, AutoModelForCausalLM
from ra_rope_config import RARopeCompressConfig
from ra_rope_tools import RARopeCompressor
import torch


def main():
    # Step 1: Load a model and tokenizer
    # Replace with your model path or HuggingFace model name
    model_name = "Qwen/Qwen3-4B"  # or "microsoft/DialoGPT-medium", etc.

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"  # Use standard attention to allow hooking
    )

    # Step 2: Create compression configuration
    # induction_head_ratio: percentage of heads to keep as induction heads
    # echo_head_ratio: percentage of heads to keep as echoing heads
    configs = [
            RARopeCompressConfig(
                induction_head_ratio=0.14,
                echo_head_ratio=0.01        # Keep top 1% echoing heads
                ),
            RARopeCompressConfig(
                induction_head_ratio=0.30,
                echo_head_ratio=0.02
                ),
            RARopeCompressConfig(
                induction_head_ratio=0.46,
                echo_head_ratio=0.04
                ),
            ]

    for i in range(len(configs)):
        f_name = f"compressible_heads_{i}.pt"

        # Step 3: Initialize compressor
        print("Initializing compressor...")
        compressor = RARopeCompressor(model, tokenizer, configs[i])

        # Step 4: Analyze model and get compressible heads
        # This will run the model on a long sequence (10,000 tokens) and
        # analyze attention patterns to identify important heads
        print("Analyzing attention patterns (this may take a while)...")
        compressor.get_compress_heads(f_name)

        print(f"done: {f_name}")



if __name__ == "__main__":
    main()
