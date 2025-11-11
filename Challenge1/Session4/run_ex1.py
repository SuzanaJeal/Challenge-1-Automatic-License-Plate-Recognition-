# -*- coding: utf-8 -*-
"""

@author: suzan
"""

# run_ex1.py
import os, glob
import numpy as np
import cv2
from LicensePlateCharacterSegmentation import detectCharacterCandidates


IMAGES_DIR = r"C:\Users\suzan\Desktop\ActividadVision\A4\Lateral"  
NPZ_PATH   = IMAGES_DIR + r"\PlateRegions.npz"


def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.files)
    print("[NPZ keys]", keys)
    regions = d['regionsImCropped']                  
    # accept either 'ImID' or 'imID'
    if 'ImID' in keys: 
        imids = d['ImID']
    elif 'imID' in keys:
        imids = d['imID']
    else:
        imids = None
    return regions, imids


def list_images(images_dir):
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.png"))) \
         + sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    return imgs

def pick_image(images_dir, imids, idx):
    imgs = list_images(images_dir)
    if imids is not None:
        im_id = str(imids[idx]).strip()
        im_id_low = im_id.lower()
        # first try strict prefix match
        cands = [p for p in imgs if os.path.basename(p).lower().startswith(im_id_low)]
        # if no hit, try substring match (common when exporters add extra tokens)
        if not cands:
            cands = [p for p in imgs if im_id_low in os.path.basename(p).lower()]
        if cands:
            print(f"[MATCH] imID '{im_id}' -> {os.path.basename(cands[0])}")
            return cands[0]
        else:
            print(f"[MISS] imID '{im_id}' not found as filename prefix/substring; falling back to index.")
    # fallback: index-based
    if idx < len(imgs):
        print(f"[NOTE] Using image by index {idx}: {os.path.basename(imgs[idx])}")
        return imgs[idx]
    return None


def main(show=True, max_items=5):
    regs, imids = load_npz(NPZ_PATH)
    for i, reg in enumerate(regs):
        img_path = pick_image(IMAGES_DIR, imids, i)
        if img_path is None:
            print(f"[WARN] No image for region index {i}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read {img_path}")
            continue

        plate, thr, cand = detectCharacterCandidates(img, reg, SHOW=0)  # teacher’s function

        # Visualize & save intermediates as the brief asks (a)
        if show:
            cv2.imshow("Plate (rectified)", plate)
            cv2.imshow("Adaptive threshold", thr)
            cv2.imshow("Character candidates (mask)", cand)
            cv2.waitKey(1)           # small non-blocking wait
            cv2.destroyAllWindows()  # close any windows


        base = os.path.splitext(os.path.basename(img_path))[0]
        outdir = os.path.join(IMAGES_DIR, "outputs_ex1")  
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(os.path.join(outdir, f"{base}_plate.png"), plate)
        cv2.imwrite(os.path.join(outdir, f"{base}_thr.png"), thr)
        cv2.imwrite(os.path.join(outdir, f"{base}_cands.png"), cand)

        if max_items and (i+1) >= max_items:
            break

if __name__ == "__main__":
    main(show=True, max_items=1)
