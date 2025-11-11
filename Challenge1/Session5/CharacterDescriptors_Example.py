# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

##### PYTHON PACKAGES
# Generic
from imutils import paths
import argparse
import pickle
import cv2
import imutils
import numpy as np
import pandas
import os
from matplotlib import pyplot as plt

# Classifiers
# include differnet classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from descriptors.blockbinarypixelsum import FeatureBlockBinaryPixelSum
from descriptors.intensity import FeatureIntensity
from descriptors.lbp import FeatureLBP
from descriptors.hog import FeatureHOG

# Helper for LBP preprocessing
def prep_for_lbp(img, double_size=True, pad=5):
    import cv2
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    if double_size:
        h, w = img.shape[:2]
        img = cv2.resize(img, (2*w, 2*h))
    return img



#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir=r'C:\Users\suzan\Desktop\ActividadVision\A5\example_fonts\example_fonts'
ResultsDir=r'C:\Users\suzan\Desktop\ActividadVision\A5\example_fonts\results'
# Load Font DataSets
fileout=os.path.join(DataDir,'alphabetIms')+'.pkl'    
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()  
alphabetIms=data['alphabetIms']
alphabetLabels=np.array(data['alphabetLabels'])
   

fileout=os.path.join(DataDir,'digitsIms')+'.pkl'   
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()   
digitsIms=data['digitsIms']
digitsLabels=np.array(data['digitsLabels'])

digitsFeat={}
alphabetFeat={}

digitsFeat['BLCK_AVG']=[]


# --- Choose a canonical preprocessing so features are comparable ---
CANON_SIZE = (40, 20)   # (H, W) after the 2× resize; use same across all images
PAD_PIXELS = 5

def prep_fixed(img):
    # pad + 2× (as earlier), then force a fixed output size
    import cv2
    img = prep_for_lbp(img, double_size=True, pad=PAD_PIXELS)
    img = cv2.resize(img, (CANON_SIZE[1], CANON_SIZE[0]))  # (W,H)
    return img

# --- Two extractors: GLOBAL vs BLOCK-WISE (both uniform for stability) ---
lbp_global = FeatureLBP(radius=3, method='uniform', lbp_type='simple')
lbp_block  = FeatureLBP(radius=3, method='uniform', lbp_type='block_lbp')  # default grid (e.g., 4×4)


# CharacterDescriptors_Example.py  :contentReference[oaicite:8]{index=8}
def prep_for_hog(img, double_size=True, pad=5):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    if double_size:
        h, w = img.shape[:2]
        img = cv2.resize(img, (2*w, 2*h))   # 2× as asked by the brief
    return img


# initialize descriptors
blockSizes =((5, 5),)#((5, 5), (5, 10), (10, 5), (10, 10))
descBlckAvg = FeatureBlockBinaryPixelSum()


### EXTRACT FEATURES
# Digits
for roi in digitsIms:
     
     # extract features
     digitsFeat['BLCK_AVG'].append(descBlckAvg.extract_image_features(roi))


### VISUALIZE FEATURE SPACES
color=['r','m','g','cyan','y','k','orange','lime','b']
from sklearn.manifold import TSNE,trustworthiness
tsne = TSNE(n_components=2, random_state=42)

for targetFeat in digitsFeat.keys():
    embeddings_2d = tsne.fit_transform(np.stack(digitsFeat[targetFeat]))
    
    plt.figure()
    plt.scatter(embeddings_2d[digitsLabels=='0', 0], embeddings_2d[digitsLabels=='0', 1], 
                marker='s')
    k=0
    for num in np.unique(digitsLabels)[1::]:
        plt.scatter(embeddings_2d[digitsLabels==num, 0], embeddings_2d[digitsLabels==num, 1], 
                     marker='o',color=color[k])
        k=k+1
    plt.legend(np.unique(digitsLabels))
    plt.title(targetFeat)
    plt.savefig(os.path.join(ResultsDir,targetFeat+'DigitsFeatSpace.png'))
    
### VISUALIZE FEATURES IMAGES




# HOG Images for Digits

# pick one example per label 0..9
sample_idx = []
seen = set()
for i, y in enumerate(digitsLabels):
    if y not in seen:
        sample_idx.append(i); seen.add(y)
    if len(seen) == 10:
        break

cell_sizes = [(4,4), (6,6), (8,8)]
for ppc in cell_sizes:
    hog_viz = FeatureHOG(pixels_per_cell=ppc, cells_per_block=(2,2),
                         orientations=9, visualize=True, feature_vector=True)
    pairs = []
    for i in sample_idx:
        img = prep_for_hog(digitsIms[i], double_size=True, pad=5)
        hog_img = hog_viz.extract_pixel_features(img)  # returns HOG "image" visualisation
        pairs.append((digitsLabels[i], img, hog_img))

    # show/save grid
    cols = len(pairs)
    fig, axs = plt.subplots(2, cols, figsize=(1.6*cols, 3.6))
    fig.suptitle(f"HOG images – pixels_per_cell={ppc}")
    for c, (lab, orig, hogim) in enumerate(pairs):
        axs[0, c].imshow(orig, cmap='gray'); axs[0, c].axis('off'); axs[0, c].set_title(str(lab))
        axs[1, c].imshow(hogim, cmap='gray'); axs[1, c].axis('off')
    plt.tight_layout()
    # plt.savefig(os.path.join(ResultsDir, f"HOG_viz_ppc_{ppc[0]}x{ppc[1]}.png"), dpi=150)  # optional
    plt.show()
    
# HOG descriptors (vectors) for all digits
descHOG = FeatureHOG(pixels_per_cell=(6,6), cells_per_block=(2,2),
                     orientations=9, visualize=False, feature_vector=True)  # vector

X = []
y = []
for img, lab in zip(digitsIms, digitsLabels):
    img2 = prep_for_hog(img, double_size=True, pad=5)   # pad+2× as brief suggests
    vec = descHOG.extract_image_features(img2)
    X.append(vec); y.append(lab)

import numpy as np
X = np.asarray(X, dtype=object)  # allow differing lengths to check
y = np.asarray(y)

# Check dimensionality
lengths = np.array([len(v) for v in X])
print("Unique HOG lengths:", np.unique(lengths))



## LBP Images for Digits
# pick one example per label 0..9
sample_idx = []
seen = set()
for i, y in enumerate(digitsLabels):
    if y not in seen:
        sample_idx.append(i); seen.add(y)
    if len(seen) == 10:
        break

radii = [1, 2, 3, 4]  # try 1..4
for r in radii:
    lbp = FeatureLBP(radius=r, method='default', lbp_type='simple')  # non-invariant
    cols = len(sample_idx)
    fig, axs = plt.subplots(2, cols, figsize=(1.6*cols, 3.6))
    fig.suptitle(f"LBP images – radius={r} (method=default)")
    for c, idx in enumerate(sample_idx):
        img = prep_for_lbp(digitsIms[idx], double_size=True, pad=5)
        lbp_img = lbp.extract_pixel_features(img)  # H×W×1 map of LBP codes
        axs[0, c].imshow(img, cmap='gray'); axs[0, c].axis('off'); axs[0, c].set_title(str(digitsLabels[idx]))
        axs[1, c].imshow(lbp_img[...,0], cmap='gray'); axs[1, c].axis('off')
    plt.tight_layout()
    # plt.savefig(os.path.join(ResultsDir, f"LBP_images_r{r}_default.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
# find a 6 and a 9
# Works for both string and int labels
i6 = next(i for i,y in enumerate(digitsLabels) if str(y) == '6')
i9 = next(i for i,y in enumerate(digitsLabels) if str(y) == '9')


img6 = prep_for_lbp(digitsIms[i6], True, 5)
img9 = prep_for_lbp(digitsIms[i9], True, 5)

for m in ['default','uniform']:
    lbp = FeatureLBP(radius=3, method=m, lbp_type='simple')   # global hist
    h6 = lbp.extract_image_features(img6)  # 1×n_bins (normalised)
    h9 = lbp.extract_image_features(img9)
    # cosine similarity as a quick measure
    sim = float(np.dot(h6, h9) / (np.linalg.norm(h6)*np.linalg.norm(h9) + 1e-9))
    print(f"LBP hist similarity (6 vs 9) – method={m}: {sim:.3f}")
    
# --- 3(c) BLOCK-WISE LBP for digits 6 and 9, then 2-D projection ---
import numpy as np, os, matplotlib.pyplot as plt
from descriptors.lbp import FeatureLBP

# 2.1 One extractor reused for all samples (uniform, block-wise)
lbp_block = FeatureLBP(radius=3, method='uniform', lbp_type='block_lbp')

# 2.2 Collect features (ensure labels work if they are strings)
X_list, y_list = [], []
for img, lab in zip(digitsIms, digitsLabels):
    if str(lab) in ('6','9'):
        img2 = prep_for_lbp(img, double_size=True, pad=5)
        feat = lbp_block.extract_image_features(img2)        # flat vector
        X_list.append(np.asarray(feat, dtype=np.float32))    # enforce float32
        y_list.append(int(lab))                               # 6 or 9 as int

# 2.3 Sanity checks (must be a single feature length)
lens = [v.shape[0] for v in X_list]
print("LBP block-wise feature lengths (unique):", sorted(set(lens)))
if len(set(lens)) != 1:
    raise ValueError("Inconsistent feature lengths; check radius/method/grid/resizing are constant.")

X = np.vstack(X_list)     # shape (N, D)
y = np.array(y_list)      # shape (N,)
print("X shape:", X.shape, "y shape:", y.shape, "counts:", {k:int((y==k).sum()) for k in [6,9]})

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 3.1 Standardize
Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

# 3.2 Safe PCA dimension: must be < min(n_samples, n_features)
max_pca = max(2, min(50, Xs.shape[1], Xs.shape[0]-1))
use_pca = max_pca >= 3
Xp = PCA(n_components=max_pca, random_state=0).fit_transform(Xs) if use_pca else Xs
print("Using PCA:", use_pca, " -> Xp shape:", Xp.shape)

# 3.3 Safe t-SNE perplexity: < ~ n_samples/3
n = Xp.shape[0]
perp = min(30, max(5, (n // 3) - 1))
print("t-SNE perplexity:", perp, " (n samples:", n, ")")

tsne = TSNE(
    n_components=2,
    perplexity=perp,
    learning_rate='auto',
    init='random',
    random_state=0,
    n_iter=1000,
    verbose=1
)
Z = tsne.fit_transform(Xp)    # shape (N, 2)

plt.figure(figsize=(6,5))
colors = {6:'tab:orange', 9:'tab:blue'}
for lab in (6,9):
    idx = (y == lab)
    plt.scatter(Z[idx,0], Z[idx,1], s=12, c=colors[lab], label=str(lab))
plt.legend()
plt.title("t-SNE — Block-wise LBP (uniform) for digits 6 vs 9")
if 'ResultsDir' in globals():
    out = os.path.join(ResultsDir, "tsne_lbp_block_6v9.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("Saved:", out)
plt.show()

    
Xg_list, Xb_list, y_list = [], [], []
for img, lab in zip(digitsIms, digitsLabels):
    img2 = prep_fixed(img)
    # GLOBAL (single histogram)
    Xg_list.append(np.asarray(lbp_global.extract_image_features(img2), dtype=np.float32))
    # BLOCK-WISE (concatenated histograms over grid)
    Xb_list.append(np.asarray(lbp_block.extract_image_features(img2), dtype=np.float32))
    y_list.append(int(lab))

# Convert to matrices and verify fixed lengths
Xg = np.vstack(Xg_list)    # shape (N, Dg)
Xb = np.vstack(Xb_list)    # shape (N, Db)
y  = np.asarray(y_list)

print("GLOBAL LBP shape:", Xg.shape)  # Dg should equal n_bins (≈ n_points+2 for uniform)
print("BLOCK  LBP shape:", Xb.shape)  # Db should equal n_bins * (grid_y * grid_x)
