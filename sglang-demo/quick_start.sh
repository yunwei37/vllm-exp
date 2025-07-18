#!/bin/bash

echo "SGLang Quick Start Demo"
echo "======================"
echo ""

echo "Step 1: Installing SGLang..."

# Check if sglang is already installed
if python -c "import sglang" 2>/dev/null; then
    echo "SGLang is already installed."
else
    echo "Installing SGLang with pip..."
    pip install --upgrade pip
    pip install uv
    uv pip install "sglang[all]>=0.4.9.post2"
fi

echo ""
echo "Step 2: Creating test script..."

# Create a simple test script  
cat > test_sglang.py << 'EOF'
import sglang as sgl

print("SGLang Demo - Frontend Language")
print("==============================")
print("")

# Example 1: Basic function definition
@sgl.function
def chat_example(s, question):
    s += "User: " + question + "\n"
    s += "Assistant: " + sgl.gen("response", max_tokens=50, temperature=0.7)

# Example 2: Multi-turn conversation
@sgl.function
def multi_turn_chat(s):
    s += "System: You are a helpful assistant.\n"
    s += "User: What is Python?\n"
    s += "Assistant: " + sgl.gen("answer1", max_tokens=50) + "\n"
    s += "User: Can you give me a simple example?\n"
    s += "Assistant: " + sgl.gen("answer2", max_tokens=100)

# Example 3: Structured generation
@sgl.function
def structured_output(s):
    s += "Generate a JSON object for a person:\n"
    s += sgl.gen("json_output", 
                 max_tokens=100,
                 regex=r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}')

print("SGLang functions defined successfully!")
print("")
print("To run these examples, you need to:")
print("1. Start the SGLang server:")
print("   ./start_server.sh")
print("")
print("2. In Python, set the backend and run:")
print("   import sglang as sgl")
print("   sgl.set_default_backend(sgl.RuntimeEndpoint('http://localhost:30000'))")
print("   state = chat_example.run(question='What is the capital of France?')")
print("   print(state['response'])")
print("")
print("3. Or use the REST API:")
print("   ./test_api.py")
EOF

echo ""
echo "Step 3: Running SGLang test..."
python test_sglang.py

echo ""
echo "Demo complete! You can now:"
echo "1. Start server: python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 0.0.0.0 --port 30000"
echo "2. Use the Python API as shown in test_sglang.py"
echo "3. Use the OpenAI-compatible REST API"