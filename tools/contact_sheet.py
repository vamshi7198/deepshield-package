"""Makes contact sheets (grids) from rendered slide PNGs. Usage: python contact_sheet.py <dir> [cols] [per_sheet]"""
import glob, os, sys
from PIL import Image, ImageDraw
d = sys.argv[1]; cols = int(sys.argv[2]) if len(sys.argv) > 2 else 2; per = int(sys.argv[3]) if len(sys.argv) > 3 else 6
fs = sorted(glob.glob(os.path.join(d, "slide-*.png")))
for k in range(0, len(fs), per):
    ims = [Image.open(f).convert("RGB") for f in fs[k:k + per]]
    tw = 1000; th = int(ims[0].height * tw / ims[0].width); rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 10, rows * th + (rows + 1) * 10), "#888")
    for i, im in enumerate(ims):
        x, y = 10 + (i % cols) * (tw + 10), 10 + (i // cols) * (th + 10)
        sheet.paste(im.resize((tw, th)), (x, y))
        ImageDraw.Draw(sheet).text((x + 6, y + 6), os.path.basename(fs[k + i]), fill="#f00")
    out = os.path.join(d, f"sheet-{k // per + 1}.jpg"); sheet.save(out, quality=85); print(out)
