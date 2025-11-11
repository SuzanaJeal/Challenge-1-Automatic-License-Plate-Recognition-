# -*- coding: utf-8 -*-
"""

@author: suzan
"""

# run_ex2.py
import os, glob
import numpy as np
import cv2
from LicensePlateCharacterSegmentation import detectCharacterCandidates

# --- External libraries for blob detection ---
from skimage.feature import blob_log          # Laplacian of Gaussian
from scipy import ndimage as ndi              # gaussian_laplace

# === EDIT THESE PATHS ===
IMAGES_DIR = r"C:\Users\suzan\Desktop\ActividadVision\A4\Frontal"
NPZ_PATH   = os.path.join(IMAGES_DIR, "PlateRegions.npz")

def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    regs = d['regionsImCropped']
    imids = d['imID'] if 'imID' in d.files else None
    return regs, imids

def list_images(images_dir):
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.png"))) \
         + sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    return imgs

def pick_image(images_dir, imids, idx):
    imgs = list_images(images_dir)
    if imids is not None:
        iid = str(imids[idx])
        c = [p for p in imgs if os.path.basename(p).startswith(iid)]
        if c: return c[0]
    return imgs[idx] if idx < len(imgs) else None

def main(limit=2, show=False, save=True):
    regs, imids = load_npz(NPZ_PATH)
    outdir = os.path.join(IMAGES_DIR, "outputs_ex2")
    os.makedirs(outdir, exist_ok=True)

    for i, reg in enumerate(regs):
        img_path = pick_image(IMAGES_DIR, imids, i)
        if not img_path: break
        img = cv2.imread(img_path)
        plate, _, _ = detectCharacterCandidates(img, reg, SHOW=0)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        base = os.path.splitext(os.path.basename(img_path))[0]

        # --- (a) skimage.feature.blob_log ---
        blobs = blob_log(gray, min_sigma=1.5, max_sigma=6.0,
                         num_sigma=10, threshold=0.02, overlap=0.2)
        vis1 = plate.copy()
        for (y, x, s) in blobs:
            r = int(s * np.sqrt(2))
            cv2.circle(vis1, (int(x), int(y)), r, (255, 0, 0), 2)

        # --- (b) scipy.ndimage.gaussian_laplace ---
        sigmas = [1.5, 2.0, 2.5, 3.0]
        responses = [-ndi.gaussian_laplace(gray.astype(float), sigma=s) for s in sigmas]
        resp = np.max(np.stack(responses, axis=0), axis=0)
        resp_n = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bw = cv2.threshold(resp_n, int(0.35*resp_n.max()), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, 1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, 1)

        # --- Save results ---
        cv2.imwrite(os.path.join(outdir, f"{base}_bloblog.png"), vis1)
        cv2.imwrite(os.path.join(outdir, f"{base}_gl_resp.png"), resp_n)
        cv2.imwrite(os.path.join(outdir, f"{base}_gl_bw.png"), bw)
        print(f"[SAVED] {base} results in {outdir}")

        if show:
            cv2.imshow("LoG blobs (skimage)", vis1)
            cv2.imshow("gaussian_laplace norm", resp_n)
            cv2.imshow("gaussian_laplace thresholded", bw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if limit and (i+1) >= limit:
            break

if __name__ == "__main__":
    main(limit=3)
