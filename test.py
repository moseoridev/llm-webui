from tqdm import tqdm
from transformers import pipeline

generator = pipeline(task="text-generation")
examples = [
        "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
        "Nine for Mortal Men, doomed to die, One for the Dark Lord on his dark throne",
    ]

for i in tqdm(range(len(examples))):
    generator(examples)
