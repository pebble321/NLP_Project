"""
Original file is located at
    https://colab.research.google.com/drive/1VSx35d0gVluVVu5tDH8mPz120W9pWF56?usp=sharing
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").to(device)

# Function to generate text for a single prompt
def generate_text(prompt, max_length=500, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Array of prompts

from google.colab import files
files.upload()

import pandas as pd
# comment out the line below to train with few-shot chain-of-thought examples
# df = pd.read_csv('fewshot.csv')
df = pd.read_csv('baseline.csv')

prompts = df['question'].tolist()
answers = df['golden'].tolist()
# # Generate responses for each prompt
responses = [generate_text(prompt) for prompt in prompts]

# Print the responses
for i, (prompt, response, answer) in enumerate(zip(prompts, responses, answers)):
    print(f"Prompt {i + 1}: {prompt}")
    print(f"golden answer {i + 1}: {answer}")
    print(f"Response {i + 1}: {response}\n")
