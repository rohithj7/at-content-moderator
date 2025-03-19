from wildguard.wildguard import WildGuardHF

import torch
if torch.backends.mps.is_available():
    device = "mps" # for apple silicon
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  # cpu in case gpu and mps are not available

print(f"Using device: {device}")

classifier = WildGuardHF(
    model_path="./content-moderator-model",  # Adjust this path as needed
    batch_size=16,
    device=device
)

items = [
    {"prompt": "I killed the game today. I was open most of the time and their DB didn't even do anything about it."}
]

results = classifier.classify(items)

for i, result in enumerate(results):
    print(f"Item {i+1}:")
    print(f"  Prompt harmfulness: {result.get('prompt_harmfulness')}")
    print()