import os
from tqdm import tqdm
import random
import re
from sentence_transformers import SentenceTransformer

random.seed(32)


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file, "r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats


def extract_feats_using_model(model, file):
    with open(file, "r") as fread:
        text = fread.read().strip()
    embeddings = model.encode([text], convert_to_tensor=True)
    print(f"{embeddings.shape=}")
    return embeddings.squeeze(0)

