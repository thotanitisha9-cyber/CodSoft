# Face Detection and Recognition System

A complete Python-based artificial intelligence system that detects and recognizes human faces in images, videos, and live webcam streams. Built with Python, OpenCV, PyTorch, and InceptionResnetV1 (FaceNet).

---

## 1. Project Overview
This project provides a robust, production-ready biometric face detection and verification pipeline. The system combines traditional Computer Vision algorithms for rapid detection with modern Deep Learning networks to extract high-fidelity facial embeddings. It offers:
- **Modular Architecture**: Separate packages for detection, feature extraction, training, orchestration, and interface.
- **Biometric Calibration**: Evaluation pipeline using Leave-One-Out cross-validation reporting Accuracy, Precision, Recall, and F1 metrics.
- **Multiple Interfaces**: Native OpenCV webcam window and an interactive, glassmorphic Streamlit web application.

---

## 2. Project Objectives
- Detect multiple human faces in real-time or static frames.
- Align and crop facial regions of interest (ROIs).
- Map face crops into a 512-dimensional vector space (embeddings).
- Compare vectors using Cosine Similarity against a serialized pickle database (`models/embeddings.pkl`).
- Draw premium bounding boxes and custom colored tags representing recognized identities (or "Unknown" if similarity is below the threshold).
- Display real-time video processing statistics like instantaneous FPS.

---

## 3. Face Detection Backends
The system supports two independent face detection backends, selectable via configuration parameters:

1. **Haar Cascade Classifier (Traditional)**:
   - Uses OpenCV's pre-trained frontal face cascade model.
   - Extremely fast and lightweight, ideal for CPUs and low-spec edge hardware.
   
2. **MTCNN (Deep Learning)**:
   - Multi-task Cascaded Convolutional Network from the `facenet-pytorch` package.
   - Leverages a three-stage cascade structure (P-Net, R-Net, O-Net) to predict bounding boxes, detection probability scores, and facial keypoints (eyes, nose, mouth corners).
   - High robustness against changes in head posture, lighting, and facial expressions.

---

## 4. Face Recognition Architecture
Face recognition is achieved using a Deep Learning-based feature extraction pipeline:

### Haar Cascade Explanation
Traditional Haar cascades operate by sliding a search window across an image and calculating **Haar-like features** (the difference between the sum of pixels in adjacent rectangular regions). 
- **Integral Images** are used to calculate these sums in $O(1)$ time.
- **Adaboost** selects a small subset of the most descriminative features.
- The features are arranged in a **Cascade Classifier** structure: sub-windows that fail early stages are immediately rejected, drastically accelerating computation.

### FaceNet (InceptionResnetV1) Explanation
FaceNet directly learns a mapping from face images to a compact Euclidean space where distances correspond to face similarity.
- **Embeddings**: A deep convolutional network (we use `InceptionResnetV1` pre-trained on `VGGFace2`) maps a cropped $160\times 160$ face image into a $512$-dimensional vector.
- **L2 Normalization**: The vector is scaled to a unit length of $1.0$ (i.e., its L2 norm is $1.0$).
- **Matching Metric (Cosine Similarity)**: For two normalized vectors $A$ and $B$, the cosine similarity is simply their dot product ($A \cdot B$). A high dot product (e.g., $\ge 0.60$) indicates matching identities.
- **Unknown Detection**: If the highest dot product against all enrolled samples in the database is below the configured threshold, the subject is flagged as `"Unknown"`.

---

## 5. Dataset Description
The database is structured as a directory tree where each subdirectory represents a unique individual.

```text
dataset/
├── Alice/
│   ├── alice_1.png
│   └── alice_2.png
├── Bob/
│   ├── bob_1.png
│   └── bob_2.png
└── Charlie/
    ├── charlie_1.png
    └── charlie_2.png
```

The system is designed to handle multiple reference images per person to capture variation in illumination, expressions, and posture.

---

## 6. Project Directory Structure
```text
├── dataset/                    # Reference images sorted into directories by name
├── models/                     # Holds serialized embedding models
│   └── embeddings.pkl
├── detect_faces.py             # Unified face detection API (Haar & MTCNN)
├── embeddings.py               # Feature extractor (FaceNet) & cosine matching logic
├── train.py                    # Scans dataset, saves embeddings, computes LOO metrics
├── recognize_faces.py          # Orchestrates detection + recognition + visualization
├── webcam.py                   # Standalone local webcam recognition application
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Package dependencies
└── README.md                   # System documentation
```

---

## 7. Technologies Used
- **Language**: Python 3.14+
- **Deep Learning**: PyTorch, torchvision, facenet-pytorch (InceptionResnetV1)
- **Computer Vision**: OpenCV (opencv-python)
- **User Interface**: Streamlit
- **Math/Data Utilities**: NumPy, Pandas, Scikit-learn, SciPy
- **Image Handling**: Pillow (PIL)

---

## 8. Installation Steps

1. **Clone the repository** (or navigate to the workspace directory):
   ```bash
   cd "c:/Users/THOTA NITISHA/OneDrive/Desktop/CodSoft-FD"
   ```

2. **Install requirements**:
   Since Python 3.14 is a modern release, install the `facenet-pytorch` package without checking compiler dependencies to prevent compilation errors of older modules:
   ```bash
   pip install -r requirements.txt --no-deps
   ```
   *Note: Standard pip package managers might attempt to build older numpy versions; installing with `--no-deps` is recommended on Python 3.14 as the necessary `numpy`, `torch`, and `torchvision` dependencies are already preinstalled.*

---

## 9. How to Run

### Option 1: Streamlit Interactive Dashboard (Recommended)
Launch the premium web application to upload images, process video files, stream webcam feeds, and manage the dataset:
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option 2: Train Model (Command Line)
To extract embeddings and generate metrics from the dataset directory:
```bash
python train.py --method mtcnn --threshold 0.60
```

### Option 3: Image Face Recognition (Command Line)
To recognize faces in a static image file:
```bash
python recognize_faces.py --image path/to/input.jpg --output output_annotated.jpg --threshold 0.60
```

### Option 4: Live Webcam stream (Command Line)
To launch a real-time OpenCV camera viewer overlay:
```bash
python webcam.py --method mtcnn --threshold 0.60
```

---

## 10. Sample Outputs

### Training and Cross-Validation
Running `python train.py` on the included dataset yields:
```text
Starting Training Phase using detection method: mtcnn...
Processing images for: Alice
Processing images for: Bob
Processing images for: Charlie

--- Training Phase Summary ---
Total images processed: 6
Faces successfully detected: 6
Embeddings saved to: models\embeddings.pkl
Classes in database: ['Alice', 'Bob', 'Charlie']

Evaluating system performance...
--- Evaluation Metrics ---
Accuracy:        1.0000
Macro Precision: 1.0000
Macro Recall:    1.0000
Macro F1 Score:  1.0000
Micro F1 Score:  1.0000
Total Eval Samples: 6
```

### CLI Recognition Output
Running `python recognize_faces.py --image dataset/Bob/bob_2.png` yields:
```text
Loaded embeddings database containing 3 classes.
Successfully processed image. Saved output to output_recognized.jpg
Results:
Face 1: Name=Bob, Similarity=1.0000, Box=[322, 205, 372, 501]
```

---

## 11. Future Enhancements
- **SQL Database Integration**: Swap pickle serialization with a PostgreSQL/SQLite Vector DB (like pgvector) to support scaling up to millions of face embeddings.
- **Anti-Spoofing (Liveness Detection)**: Implement texture-analysis or blink detection algorithms to prevent spoofing using photographs.
- **Batch Processing**: Implement asynchronous multi-threading to speed up large video processing batches.
- **Alternative Embeddings**: Integrate ArcFace / RetinaFace backends.
