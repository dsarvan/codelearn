from PIL import Image
import os
pil_im = Image.open("/home/saran/Pictures/wall.jpg")
pil_im = Image.open("/home/saran/Pictures/wall.jpg").convert("L")
