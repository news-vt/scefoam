from PIL import Image, ImageDraw
import numpy as np

# Paths
input_path = "/mnt/data/frame_5_14s.png"
output_path = "/mnt/data/partial_block_packet_loss.png"

# Load the frame
img = Image.open(input_path).convert("RGB")
w, h = img.size
draw = ImageDraw.Draw(img)

# Define macroblock size and drop probability
block_size = 128
drop_prob  = 0.5   # 50% chance to drop each block

# Iterate over the frame in block_size increments
for y in range(0, h, block_size):
    for x in range(0, w, block_size):
        if np.random.rand() < drop_prob:
            draw.rectangle([x, y, x+block_size, y+block_size], fill=(0,0,0))

# Save the result
img.save(output_path)

output_path
