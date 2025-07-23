#!/usr/bin/env python3
"""
vLLM Quick Start Example
This script demonstrates basic offline batch inference with vLLM
"""

from vllm import LLM, SamplingParams

# Define test prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Set sampling parameters
# temperature: controls randomness (0.0 = deterministic, 1.0 = more random)
# top_p: nucleus sampling threshold
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the model
# Using a small model (facebook/opt-125m) for quick testing
print("Loading model...")
llm = LLM(model="facebook/opt-125m")

# Generate outputs
print("Generating responses...")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs
print("\n" + "="*50 + "\n")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print("-" * 50)