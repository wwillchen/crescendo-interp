import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it", dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
print(f"Model class: {type(model).__name__}")
print(f"Top-level children:")
for name, child in model.named_children():
    print(f"  {name}: {type(child).__name__}")
    for name2, child2 in child.named_children():
        print(f"    {name2}: {type(child2).__name__}")
        if hasattr(child2, '__len__'):
            print(f"      len={len(child2)}")
        for name3, child3 in list(child2.named_children())[:3]:
            print(f"      {name3}: {type(child3).__name__}")
