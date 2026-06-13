import os
import cv2
import numpy as np
from PIL import Image

# Global variables for caching detectors
_HAAR_CASCADE = None
_MTCNN_DETECTOR = None

def get_haar_cascade():
    """Loads and caches the Haar Cascade Classifier."""
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _HAAR_CASCADE

def get_mtcnn_detector():
    """Loads and caches the MTCNN detector from facenet-pytorch."""
    global _MTCNN_DETECTOR
    if _MTCNN_DETECTOR is None:
        try:
            from facenet_pytorch import MTCNN
            # MTCNN works best when initialized with a fixed image size or keep_all=True to find all faces.
            # We want keep_all=True so it can detect multiple faces.
            _MTCNN_DETECTOR = MTCNN(keep_all=True, device='cpu')
        except ImportError:
            print("Warning: facenet_pytorch not installed or MTCNN failed to import. Falling back to Haar Cascade.")
            _MTCNN_DETECTOR = None
    return _MTCNN_DETECTOR

def detect_faces(image, method='mtcnn', confidence_threshold=0.5):
    """
    Detects faces in an image using the specified method ('mtcnn' or 'haar').
    
    Parameters:
        image (np.ndarray or PIL.Image.Image): Input image. If numpy array, expected in BGR (OpenCV default) or RGB.
        method (str): 'mtcnn' or 'haar'.
        confidence_threshold (float): Only keep detections above this confidence (MTCNN only).
        
    Returns:
        list of dict: Each dict contains:
            - 'box': [x, y, w, h] (bounding box coordinates)
            - 'confidence': float (detection confidence)
            - 'face_img': np.ndarray (RGB cropped face image)
    """
    # 1. Standardize image input
    if isinstance(image, Image.Image):
        # Convert PIL to NumPy RGB
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # By default, we assume OpenCV image is BGR
        img_bgr = image.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Image must be a numpy array or a PIL Image")

    h_img, w_img, _ = img_rgb.shape
    detections = []

    if method.lower() == 'mtcnn':
        detector = get_mtcnn_detector()
        if detector is None:
            # Fallback to Haar Cascade
            method = 'haar'
        else:
            # MTCNN expects a PIL Image or RGB numpy array. Let's pass a PIL Image.
            pil_img = Image.fromarray(img_rgb)
            boxes, probs = detector.detect(pil_img)
            
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob >= confidence_threshold:
                        # Box is [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box
                        # Ensure coordinates are within image boundaries
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(w_img, int(x2))
                        y2 = min(h_img, int(y2))
                        
                        w = x2 - x1
                        h = y2 - y1
                        
                        if w > 10 and h > 10:  # Ignore tiny detections
                            face_crop = img_rgb[y1:y2, x1:x2]
                            detections.append({
                                'box': [x1, y1, w, h],
                                'confidence': float(prob),
                                'face_img': face_crop
                            })

    if method.lower() == 'haar':
        cascade = get_haar_cascade()
        # Haar Cascade works on grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Ensure coordinates are within image boundaries
            x_clean = max(0, int(x))
            y_clean = max(0, int(y))
            w_clean = min(w_img - x_clean, int(w))
            h_clean = min(h_img - y_clean, int(h))
            
            face_crop = img_rgb[y_clean:y_clean+h_clean, x_clean:x_clean+w_clean]
            detections.append({
                'box': [x_clean, y_clean, w_clean, h_clean],
                'confidence': 1.0,  # Haar Cascade doesn't give a standard probability score
                'face_img': face_crop
            })

    return detections

if __name__ == "__main__":
    # Small test
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    print("Testing Face Detection Module...")
    try:
        dets = detect_faces(test_img, method='haar')
        print(f"Haar Cascade works! Found {len(dets)} faces in blank image.")
        dets_mtcnn = detect_faces(test_img, method='mtcnn')
        print(f"MTCNN works! Found {len(dets_mtcnn)} faces in blank image.")
    except Exception as e:
        print(f"Error occurred during test: {e}")
