
from PIL import Image
import numpy as np
import os

def lsb_embed(input_path, output_path, message="secret"):
    img = Image.open(input_path).convert("RGB")
    data = np.array(img)

    binary = ''.join(format(ord(c), '08b') for c in message) + '11111111'
    idx = 0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(3):
                if idx < len(binary):
                    data[i, j, k] = (data[i, j, k] & 0xFE) | int(binary[idx])
                    idx += 1

    Image.fromarray(data).save(output_path)

os.makedirs("dataset/stego", exist_ok=True)
for img in os.listdir("dataset/clean"):
    lsb_embed(f"dataset/clean/{img}", f"dataset/stego/{img}")
