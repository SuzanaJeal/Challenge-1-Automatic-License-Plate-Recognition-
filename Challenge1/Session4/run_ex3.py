# -*- coding: utf-8 -*-
"""

@author: suzan
"""

# run_ex3.py
import os, glob
import numpy as np
import cv2
from LicensePlateCharacterSegmentation import detectCharacterCandidates

# ==== EDIT THIS ====
IMAGES_DIR = r"C:\Users\suzan\Desktop\ActividadVision\A4\Frontal"
NPZ_PATH   = os.path.join(IMAGES_DIR, "PlateRegions.npz")


def crop_inner_plate(plate_bgr):
    H, W = plate_bgr.shape[:2]

    # 1) trim a thin frame all around (plate border)
    x_margin = int(0.04 * W)   # ~4% each side
    y_margin = int(0.12 * H)   # ~12% top/bottom (often plate has more vertical border)
    x0, x1 = x_margin, W - x_margin
    y0, y1 = y_margin, H - y_margin
    roi = plate_bgr[max(0,y0):min(H,y1), max(0,x0):min(W,x1)]

    # 2) detect & strip EU blue band on the left if present
    #    check the first 15% width for strong blue
    H2, W2 = roi.shape[:2]
    probe_w = max(4, int(0.15 * W2))
    probe = roi[:, :probe_w]
    hsv = cv2.cvtColor(probe, cv2.COLOR_BGR2HSV)
    # blue mask (rough): H∈[90,135], S>80, V>50
    blue = cv2.inRange(hsv, (90, 80, 50), (135, 255, 255))
    frac_blue = (blue > 0).mean()

    if frac_blue > 0.15:
        # remove the band area (keep the rest)
        roi = roi[:, probe_w:]

    return roi


def load_npz(p):
    d = np.load(p, allow_pickle=True)
    regs = d['regionsImCropped']
    imids = d['imID'] if 'imID' in d.files else (d['ImID'] if 'ImID' in d.files else None)
    return regs, imids

def list_images(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png"))) + \
           sorted(glob.glob(os.path.join(folder, "*.jpg")))

def pick_image(folder, imids, idx):
    imgs = list_images(folder)
    if imids is not None:
        iid = str(imids[idx])
        cand = [p for p in imgs if os.path.basename(p).startswith(iid)]
        if cand: return cand[0]
    return imgs[idx] if idx < len(imgs) else None

def watershed_chars(gray):
    # --- pre-clean the image ---
    gray_blur = cv2.medianBlur(gray, 3)
    closed = cv2.morphologyEx(
        gray_blur, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

    # --- threshold (Otsu, inverted) ---
    _, binary = cv2.threshold(closed, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- morphological cleanup ---
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- sure background and foreground ---
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.60 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # --- markers ---
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # --- watershed ---
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    char_mask = (markers > 1).astype(np.uint8) * 255
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary, opening, dist, sure_fg, unknown, markers, char_mask


def main(limit=2, save=True, show=False):
    regs, imids = load_npz(NPZ_PATH)
    outdir = os.path.join(IMAGES_DIR, "outputs_ex3"); os.makedirs(outdir, exist_ok=True)

    for i, reg in enumerate(regs):
        img_path = pick_image(IMAGES_DIR, imids, i)
        if not img_path: break

        img = cv2.imread(img_path)
        # Use teacher's function to rectify the plate
        plate, _, _ = detectCharacterCandidates(img, reg, SHOW=0)
        plate = crop_inner_plate(plate)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        binary, opening, dist, sure_fg, unknown, markers, char_mask = watershed_chars(gray)

        base = os.path.splitext(os.path.basename(img_path))[0]

        if save:
            cv2.imwrite(os.path.join(outdir, f"{base}_plate.png"), plate)
            cv2.imwrite(os.path.join(outdir, f"{base}_bin.png"), binary)
            cv2.imwrite(os.path.join(outdir, f"{base}_open.png"), opening)
            cv2.imwrite(os.path.join(outdir, f"{base}_dist.png"),
                        cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            cv2.imwrite(os.path.join(outdir, f"{base}_surefg.png"), sure_fg)
            cv2.imwrite(os.path.join(outdir, f"{base}_unknown.png"), unknown)
            cv2.imwrite(os.path.join(outdir, f"{base}_chars.png"), char_mask)

            # Crop characters from char_mask
            chars_dir = os.path.join(outdir, "chars"); os.makedirs(chars_dir, exist_ok=True)
            cnts, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            H, W = char_mask.shape[:2]
            boxes = []
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                h_ratio = h / float(H)
                ar = w / float(h)
                if 0.45 <= h_ratio <= 0.95 and 0.15 <= ar <= 0.8 and area > 150:
                    if x > 0 and y > 0 and (x + w) < W and (y + h) < H:
                        boxes.append((x, y, w, h))
            boxes.sort(key=lambda b: b[0])

            for j, (x,y,w,h) in enumerate(boxes, 1):
                cv2.imwrite(os.path.join(chars_dir, f"{base}_char{j}.png"), plate[y:y+h, x:x+w])

        if show:
            cv2.imshow("Plate", plate)
            cv2.imshow("Binary", binary)
            cv2.imshow("Opening", opening)
            cv2.imshow("Characters (mask)", char_mask)
            cv2.waitKey(0); cv2.destroyAllWindows()

        if limit and (i+1) >= limit: break

if __name__ == "__main__":
    main(limit=3, save=True, show=False)

