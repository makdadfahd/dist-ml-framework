import os
from PIL import Image, ImageDraw, ImageFont

downloads_dir = os.path.expanduser('~/Downloads')

# --- 1. GENERATE TITLED BENCHMARKS (No Padding) ---
img1_path = os.path.join(downloads_dir, 'mine_vspytorch_linkedin.png')
img2_path = os.path.join(downloads_dir, 'optimizer_benchmark_linkedin.png') # adjust to .jpg if needed

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

width = max(img1.width, img2.width)
img1_resized = img1.resize((width, int(img1.height * width / img1.width)))
img2_resized = img2.resize((width, int(img2.height * width / img2.width)))

banner_height = int(width * 0.08)
font_size = int(width * 0.045)

font_paths = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
]
font = next((ImageFont.truetype(p, font_size) for p in font_paths if os.path.exists(p)), ImageFont.load_default())

total_height = (banner_height * 2) + img1_resized.height + img2_resized.height
benchmarks = Image.new('RGB', (width, total_height), color='#ffffff')
draw = ImageDraw.Draw(benchmarks)

def draw_banner(draw, text, y_offset, width, banner_height, font):
    draw.rectangle([0, y_offset, width, y_offset + banner_height], fill='#1e293b')
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = y_offset + (banner_height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill='#ffffff', font=font)

draw_banner(draw, "My Engine vs. PyTorch", 0, width, banner_height, font)
benchmarks.paste(img1_resized, (0, banner_height))
y_sec2 = banner_height + img1_resized.height
draw_banner(draw, "Optimizer Benchmarks", y_sec2, width, banner_height, font)
benchmarks.paste(img2_resized, (0, y_sec2 + banner_height))

benchmarks.save(os.path.join(downloads_dir, 'linkedin_benchmarks_clean.png'))


# --- 2. RESTRUCTURE MNIST INTO A TALL VERTICAL GRID ---
mnist_path = os.path.join(downloads_dir, 'mnist_benchmark_linkedin.png')
mnist_img = Image.open(mnist_path)

# Original layout in image: 7 columns, 4 rows (28 digits total)
orig_cols = 7
orig_rows = 4
cell_w = mnist_img.width // orig_cols
cell_h = mnist_img.height // orig_rows

# Extract all 28 digit tiles
tiles = []
for r in range(orig_rows):
    for c in range(orig_cols):
        left = c * cell_w
        top = r * cell_h
        right = left + cell_w
        bottom = top + cell_h
        tiles.append(mnist_img.crop((left, top, right, bottom)))

# Rearrange into a TALL 3-column x 5-row grid (15 digits total)
new_cols = 3
new_rows = 5
num_digits = new_cols * new_rows

tall_mnist_w = new_cols * cell_w
tall_mnist_h = new_rows * cell_h

# Add a dark title header to MNIST as well
mnist_banner_h = int(tall_mnist_w * 0.08)
mnist_font_size = int(tall_mnist_w * 0.045)
mnist_font = next((ImageFont.truetype(p, mnist_font_size) for p in font_paths if os.path.exists(p)), ImageFont.load_default())

tall_mnist = Image.new('RGB', (tall_mnist_w, tall_mnist_h + mnist_banner_h), color='#000000')
draw_mnist = ImageDraw.Draw(tall_mnist)

# Add banner
draw_banner(draw_mnist, "MNIST Predictions", 0, tall_mnist_w, mnist_banner_h, mnist_font)

# Paste tiles into the new tall grid
for i in range(num_digits):
    r = i // new_cols
    c = i % new_cols
    x = c * cell_w
    y = mnist_banner_h + (r * cell_h)
    tall_mnist.paste(tiles[i], (x, y))

tall_mnist.save(os.path.join(downloads_dir, 'linkedin_mnist_clean.png'))

print("Done! Generated 'linkedin_benchmarks_clean.png' and 'linkedin_mnist_clean.png'.")