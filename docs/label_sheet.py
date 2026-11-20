"""Draw preset names under the spheres of docs/render_shaders.py output (system python with Pillow):  python3 docs/label_sheet.py in.png [out.png]"""
import sys
from PIL import Image, ImageDraw, ImageFont
NAMES = ["Water", "Sea water", "Glass", "Oil", "Honey", "Syrup", "Milk / cream", "Wax", "Lava / magma", "Mercury", "Molten metal", "Slime / goo", "Blood", "Ice",
         "Crystal", "Smoke", "Steam", "Soot", "Flame", "Plasma", "Gas cloud"]
cols, sp = 7, 1.15
im = Image.open(sys.argv[1]).convert("RGB")
rows = (len(NAMES) + cols - 1) // cols
dr = ImageDraw.Draw(im)
try:
    f = ImageFont.truetype("DejaVuSans.ttf", max(12, im.width // 90))
except Exception:
    f = ImageFont.load_default()
s = im.width / (cols * sp + 0.3)
for i, t in enumerate(NAMES):
    r, c = divmod(i, cols)
    px = im.width / 2 + (c - (cols - 1) / 2) * sp * s
    py = im.height / 2 + (r - (rows - 1) / 2) * 1.15 * s + 0.5 * s + 4
    w = dr.textlength(t, font=f)
    dr.rectangle([px - w / 2 - 4, py - 2, px + w / 2 + 4, py + f.size + 4], fill=(0, 0, 0))
    dr.text((px - w / 2, py), t, fill=(255, 255, 255), font=f)
im.save(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
