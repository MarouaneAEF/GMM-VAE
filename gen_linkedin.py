#!/usr/bin/env python3
"""Generate a polished LinkedIn banner showing training progression."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Large comparison images saved during training (orig left, recon right)
# Each is a vertical strip of 10 images (5 orig + 5 recon side-by-side)
SNAPS = [
    ("results/gmvae_cifar10_K10/reconstructions/large_comparisons/large_comparison_epoch_10.png",  "Epoch 10"),
    ("results/gmvae_cifar10_K10/reconstructions/large_comparisons/large_comparison_epoch_30.png",  "Epoch 30"),
    ("results/gmvae_cifar10_K10/reconstructions/large_comparisons/large_comparison_epoch_60.png",  "Epoch 60"),
    ("results/gmvae_cifar10_K10/reconstructions/large_comparisons/large_comparison_epoch_100.png", "Epoch 100"),
]

N_ROWS = 5           # how many image-rows to show per column
IMG_CELL = 80        # display size of each individual image (px)
PAD = 6              # gap between cells
LABEL_H = 34         # height of column header
TITLE_H = 56         # height of the top banner
FOOTER_H = 30        # height of the bottom footer
BG = (18, 18, 28)    # dark background
ACCENT = (100, 149, 237)  # cornflower blue

def try_font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    os.makedirs("animations", exist_ok=True)

    # ── load and parse the comparison strips ──────────────────────────────────
    # Each strip is W×H where W = 2*orig_cell and H = N * orig_cell.
    # We only need the RIGHT half (reconstructions) plus the LEFT half of the
    # first strip (originals).
    snaps = []
    for path, label in SNAPS:
        img = Image.open(path).convert("RGB")
        snaps.append((img, label))

    strip_w, strip_h = snaps[0][0].size
    half_w = strip_w // 2
    orig_cell_h = strip_h // (strip_h // half_w)   # roughly square cells
    # Recompute: each cell in the strip is half_w × (strip_h/N_total_rows)
    n_total = round(strip_h / half_w)
    cell_h = strip_h // n_total   # actual pixel height per row

    def extract_col(strip_img, side, rows):
        """side='left'|'right', returns list of PIL Images (one per row)."""
        x0 = 0 if side == 'left' else half_w
        imgs = []
        for r in range(rows):
            y0 = r * cell_h
            imgs.append(strip_img.crop((x0, y0, x0 + half_w, y0 + cell_h)))
        return imgs

    originals = extract_col(snaps[0][0], 'left', N_ROWS)
    recons_per_snap = [extract_col(s, 'right', N_ROWS) for s, _ in snaps]

    # ── layout ─────────────────────────────────────────────────────────────────
    n_cols = 1 + len(snaps)   # originals + 4 epoch columns
    col_w = N_ROWS * (IMG_CELL + PAD) - PAD   # width of one column (images side by side)
    # Re-layout: stack images VERTICALLY per column
    col_w = IMG_CELL
    col_content_h = N_ROWS * (IMG_CELL + PAD) - PAD

    total_w = n_cols * (col_w + PAD * 3) + PAD * 8
    total_h = TITLE_H + LABEL_H + col_content_h + FOOTER_H + PAD * 4

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    # fonts
    font_title  = try_font(18)
    font_label  = try_font(14)
    font_footer = try_font(11)

    # ── title ──────────────────────────────────────────────────────────────────
    title = "GMM-VAE — Training Progression on CIFAR-10  (K = 10 clusters)"
    draw.text((total_w // 2, TITLE_H // 2), title,
              fill=(230, 230, 245), font=font_title, anchor="mm")

    # thin separator line
    y_sep = TITLE_H
    draw.line([(PAD * 4, y_sep), (total_w - PAD * 4, y_sep)], fill=ACCENT, width=1)

    # ── columns ────────────────────────────────────────────────────────────────
    all_cols = [("Original", originals)] + \
               [(label, recons) for (_, label), recons in zip(snaps, recons_per_snap)]

    y_label = TITLE_H + PAD
    y_images = y_label + LABEL_H

    for c, (label, imgs) in enumerate(all_cols):
        x_col = PAD * 2 + c * (col_w + PAD * 3)

        # accent bar under the label for non-original columns
        if c > 0:
            draw.rectangle([(x_col, y_label + LABEL_H - 3),
                             (x_col + col_w, y_label + LABEL_H - 1)],
                            fill=ACCENT)

        # column label (centred)
        draw.text((x_col + col_w // 2, y_label + LABEL_H // 2 - 2),
                  label, fill=(200, 210, 240) if c > 0 else (180, 195, 210),
                  font=font_label, anchor="mm")

        # images stacked vertically
        for r, img in enumerate(imgs):
            resized = img.resize((IMG_CELL, IMG_CELL), Image.LANCZOS)
            y_img = y_images + r * (IMG_CELL + PAD)
            canvas.paste(resized, (x_col, y_img))

        # thin vertical separator after this column (except last)
        if c < n_cols - 1:
            x_sep = x_col + col_w + PAD + 1
            draw.line([(x_sep, y_images - PAD // 2),
                       (x_sep, y_images + col_content_h + PAD // 2)],
                      fill=(50, 55, 75), width=1)

    # ── footer ─────────────────────────────────────────────────────────────────
    y_footer = total_h - FOOTER_H + 6
    footer = "MSE reconstruction loss  ·  KL annealing (smoothstep)  ·  Cosine LR  ·  PyTorch / Apple MPS"
    draw.text((total_w // 2, y_footer), footer,
              fill=(100, 105, 130), font=font_footer, anchor="mm")

    # ── save ───────────────────────────────────────────────────────────────────
    out = "animations/linkedin_progression.png"
    canvas.save(out, optimize=True)
    print(f"Saved → {out}  ({total_w}×{total_h} px)")


if __name__ == "__main__":
    main()
