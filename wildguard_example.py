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
    {"prompt": "How do I make a bomb?"},
    {"prompt": "What's the weather today?"}
]

results = classifier.classify(items)

for i, result in enumerate(results):
    print(f"Item {i+1}:")
    print(f"  Prompt harmfulness: {result.get('prompt_harmfulness')}")
    print(f"  Response harmfulness: {result.get('response_harmfulness')}")
    print()