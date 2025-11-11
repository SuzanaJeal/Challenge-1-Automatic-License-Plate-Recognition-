import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === 1. SETUP ===
DataDir = r'C:\Users\indre\.spyder-py3\PROJECTS\Vision_and_Learning\all_real_plates\with_Protocol'
Views = ['Frontal', 'Lateral']

# === 2. SELECT MODEL ===
#model = YOLO("yolov8n.pt")        # YOLO trained on COCO dataset
model = YOLO("LP-detection.pt")     # YOLO trained for license plates

model_classes = np.array(list(model.names.values()))
device = model.device
print(f"Model loaded on: {device}")
print(f"Classes: {model_classes}")

# === 3. CREATE RESULTS FOLDER ===
results_root = os.path.join(DataDir, "results")
os.makedirs(results_root, exist_ok=True)

# === 4. DETECTION & DATA COLLECTION ===
yoloConf = {}
yoloObj = {}

for View in Views:
    ImageFiles = sorted(glob.glob(os.path.join(DataDir, View, '*.jpg')))

    # create results subfolder
    save_dir = os.path.join(results_root, View)
    os.makedirs(save_dir, exist_ok=True)

    yoloConf[View] = []
    yoloObj[View] = []

    print(f"\nProcessing view: {View} ({len(ImageFiles)} images shown)")

    for idx, imagePath in enumerate(ImageFiles):
        # Load image
        image = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        # Extract detections
        boxes = results[0].boxes
        obj_classes = boxes.cls.cpu().numpy().astype(int)
        conf_scores = boxes.conf.cpu().numpy()
        
        # Append detections for statistics
        yoloObj[View].extend(obj_classes)
        yoloConf[View].extend(conf_scores)

        # Save annotated result ONLY for first 10 images
        if idx < 10:
            annotated = results[0].plot()
            scale = 0.5
            annotated_small = cv2.resize(
                annotated, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            save_path = os.path.join(save_dir, f"{View}_{idx+1}_detected.jpg")
            cv2.imwrite(save_path, annotated_small)

print("\nAll processed images are saved in:", results_root)

# === 5. DISTRIBUTION ANALYSIS & SAVE PLOTS ===
for View in Views:
    objs = np.array(yoloObj[View])
    confs = np.array(yoloConf[View])

    if len(objs) == 0:
        print(f"No objects detected in {View} view.")
        continue

    labels = [model_classes[o] for o in objs]

    save_dir = os.path.join(results_root, View)

    # Histogram of detected classes
    from collections import Counter
    
    # Count frequencies of detected labels
    counts = Counter(labels)
    classes = list(counts.keys())
    frequencies = list(counts.values())
    
    plt.figure(figsize=(8,4))
    plt.bar(classes, frequencies, width=0.6, color='steelblue', edgecolor='black')
    
    plt.title(f"Detected Object Distribution - {View}")
    plt.ylabel("Frequency")
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f"{View}_class_histogram.png"))
    plt.show()

    # Histogram of confidence
    plt.figure(figsize=(6,4))
    plt.hist(confs, bins=20, color='orange', alpha=0.7)
    plt.title(f"Confidence Distribution - {View}")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{View}_confidence_histogram.png"))
    plt.show()

    # Boxplot of confidence
    plt.figure(figsize=(5,4))
    plt.boxplot(confs)
    plt.title(f"Confidence Boxplot - {View}")
    plt.ylabel("Confidence Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{View}_confidence_boxplot.png"))
    plt.show()
    
    
# === 6.  MISSING-RELEVANT-DETECTION CHECK & PLOT ===
# Decide which classes are "relevant" based on the model in use
names = list(model.names.values())
if 'single_number_plate' in names:   # LP-detection.pt
    relevant_labels = {'single_number_plate', 'double_number_plate'}
    title_suffix = ' (Plate-relevant)'
else:                                # COCO model
    relevant_labels = {'car', 'truck', 'bus'}
    title_suffix = ' (Vehicle-relevant: car/truck/bus)'

# We will recompute per-image flags here (fast), or you can store them during the main loop
missing_flags = {v: [] for v in Views}  # True = no relevant detection

for View in Views:
    ImageFiles = sorted(glob.glob(os.path.join(DataDir, View, '*.jpg')))
    for imagePath in ImageFiles:
        image = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        boxes = results[0].boxes
        if len(boxes) == 0:
            missing_flags[View].append(True)
            continue
        # map cls -> label names and check if any is relevant
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        labels  = [model.names[c] for c in cls_ids]
        has_relevant = any(lbl in relevant_labels for lbl in labels)
        missing_flags[View].append(not has_relevant)

# ---- Plot (one figure with two bars per view) ----
for View in Views:
    flags = np.array(missing_flags[View])
    total = len(flags)
    miss = int(flags.sum())
    ok   = total - miss

    plt.figure(figsize=(6,4))
    plt.bar(['No relevant detection', '≥1 relevant detection'],
            [miss, ok], color=['#e57373','#81c784'], edgecolor='black')
    plt.title(f'Missing Relevant Detections — {View}{title_suffix}')
    plt.ylabel('Images')
    # add percentages on top
    for x, v in enumerate([miss, ok]):
        pct = (v/total*100) if total>0 else 0
        plt.text(x, v + max(1, total*0.02), f'{v} ({pct:.1f}%)', ha='center')
    plt.ylim(0, max(1, int(total*1.15)))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # save next to your other plots
    save_dir = os.path.join(results_root, View)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{View}_missing_relevant_detections.png'))
    plt.show()