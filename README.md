Authors and Co-creators:
Suzana Jeal and
Valentina Indrei 

🧠 Challenge #1 – Automatic License Plate Recognition (ALPR)

This project was developed as part of the Vision & Learning course at the Universitat Autònoma de Barcelona.
The goal of the challenge is to design and implement a complete computer vision pipeline capable of automatically detecting and recognizing vehicle license plates from digital images.

🚗 Overview

Automatic License Plate Recognition (ALPR) is a classic computer vision problem widely used in traffic monitoring, parking management, and security systems.
In this challenge, we explore both traditional image processing and deep learning techniques to build a full end-to-end recognition system.

🔄 Pipeline

Image Acquisition
Capture or collect vehicle images under varying conditions (viewpoint, illumination, blur, glare).
Analyze how acquisition quality affects recognition performance.

Detection / Localization
Identify and extract the region containing the license plate using:

Traditional methods (edge detection, color/texture analysis, morphological operations).

Deep learning models such as YOLO for plate detection.

Character Segmentation
Rectify the plate region and isolate each character using affine transformations, contour detection, or connected-component analysis.

Recognition
Recognize segmented characters using:

Tesseract OCR, or

Custom feature extraction (HOG, LBP) combined with classifiers (SVM, Neural Networks).

Evaluation & Validation
Compare system performance using simple splits vs. cross-validation.
Assess robustness under different acquisition and lighting conditions.

🎯 Learning Objectives

Understand the impact of image acquisition and dataset distribution.

Apply morphological operations in pre- and post-processing.

Implement object detection with YOLO.

Analyze image structures (edges, blobs, segmentation).

Extract and use image descriptors (HOG, LBP).

Evaluate model performance using proper validation techniques.

🧰 Tools & Libraries

Developed in Python using:

OpenCV

NumPy / Pandas

scikit-learn

PyTorch

Tesseract OCR

Spyder IDE (no Jupyter Notebooks)

📁 Dataset

The dataset consists of real and synthetic vehicle images containing license plates.
Students were also required to expand it by capturing new photos under diverse conditions (lighting, angle, reflections, blur) to test generalization and robustness.

✅ Outcome

By the end of the project, the resulting system can:

Detect and crop license plates from vehicle images.

Segment individual characters.

Recognize and output the full plate number as text.
