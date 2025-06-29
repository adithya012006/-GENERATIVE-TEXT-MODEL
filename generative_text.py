from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

 
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

 
prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a village",
    "Machine learning in healthcare can"
]

 
for p in prompts:
    print(f"\nPrompt: {p}")
    print(generate_text(p))
