import os
import pickle
import cv2
import numpy as np
from PIL import Image

from detect_faces import detect_faces
from embeddings import get_face_embedding, match_embedding

# Global cached database
_LOADED_DATABASE = None
_DATABASE_PATH = None

def load_embeddings_database(embeddings_file='models/embeddings.pkl'):
    """Loads and caches the face embeddings database."""
    global _LOADED_DATABASE, _DATABASE_PATH
    if _LOADED_DATABASE is None or _DATABASE_PATH != embeddings_file:
        if not os.path.exists(embeddings_file):
            print(f"Warning: Database file '{embeddings_file}' not found. Please train the system first.")
            return {}
        with open(embeddings_file, 'rb') as f:
            _LOADED_DATABASE = pickle.load(f)
        _DATABASE_PATH = embeddings_file
        print(f"Loaded embeddings database containing {len(_LOADED_DATABASE)} classes.")
    return _LOADED_DATABASE

def clear_database_cache():
    """Clears the cached database so it can be reloaded after training."""
    global _LOADED_DATABASE
    _LOADED_DATABASE = None

def recognize_faces_in_image(image, database_path='models/embeddings.pkl', detection_method='mtcnn', similarity_threshold=0.60):
    """
    Detects and recognizes faces in an image.
    
    Parameters:
        image (np.ndarray or PIL.Image.Image): Input image (BGR numpy array or PIL Image).
        database_path (str): Path to saved embeddings.
        detection_method (str): 'mtcnn' or 'haar'.
        similarity_threshold (float): Match threshold.
        
    Returns:
        tuple: (annotated_image_bgr, list of results dicts)
            where results dicts contain: 'box', 'name', 'confidence', 'similarity'
    """
    # 1. Standardize image input
    if isinstance(image, Image.Image):
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img_bgr = image.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Image must be a numpy array or a PIL Image")
        
    # Load database
    database = load_embeddings_database(database_path)
    
    # Detect faces
    detections = detect_faces(img_rgb, method=detection_method)
    
    results = []
    
    # Premium color palette (BGR format)
    # Recognized: Vibrant Teal/Mint (0x33FFB5 -> BGR: 181, 255, 51)
    # Unknown: Premium Coral/Rose (0xFA5252 -> BGR: 82, 82, 250)
    color_recognized = (181, 255, 51)
    color_unknown = (82, 82, 250)
    
    for det in detections:
        box = det['box']  # [x, y, w, h]
        face_crop = det['face_img']
        det_conf = det['confidence']
        
        x, y, w, h = box
        
        name = "Unknown"
        sim_score = 0.0
        
        if database:
            try:
                # Generate embedding
                query_emb = get_face_embedding(face_crop)
                # Match against database
                name, sim_score = match_embedding(query_emb, database, threshold=similarity_threshold)
            except Exception as e:
                print(f"Error generating embedding for face at {box}: {e}")
                
        results.append({
            'box': box,
            'name': name,
            'detection_confidence': det_conf,
            'similarity': sim_score
        })
        
        # Select color based on recognition result
        color = color_recognized if name != "Unknown" else color_unknown
        
        # Draw bounding box with rounded corners (slightly polished)
        thickness = 2
        # Main rectangle
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness)
        
        # Premium design: Draw double lines or small corners to make it look futuristic
        corner_len = min(w, h) // 5
        cv2.line(img_bgr, (x, y), (x + corner_len, y), color, thickness + 2)
        cv2.line(img_bgr, (x, y), (x, y + corner_len), color, thickness + 2)
        
        cv2.line(img_bgr, (x + w, y), (x + w - corner_len, y), color, thickness + 2)
        cv2.line(img_bgr, (x + w, y), (x + w, y + corner_len), color, thickness + 2)
        
        cv2.line(img_bgr, (x, y + h), (x + corner_len, y + h), color, thickness + 2)
        cv2.line(img_bgr, (x, y + h), (x, y + h - corner_len), color, thickness + 2)
        
        cv2.line(img_bgr, (x + w, y + h), (x + w - corner_len, y + h), color, thickness + 2)
        cv2.line(img_bgr, (x + w, y + h), (x + w, y + h - corner_len), color, thickness + 2)

        # Label drawing
        # Create text label
        if name != "Unknown":
            label = f"{name} ({sim_score:.2%})"
        else:
            label = f"Unknown"
            
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Calculate size of text to position background filled box
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Position label above box if space permits, otherwise below
        if y - text_h - 15 > 0:
            label_y1 = y - text_h - 10
            label_y2 = y
        else:
            label_y1 = y + h
            label_y2 = y + h + text_h + 10
            
        # Draw background filled block for text
        cv2.rectangle(img_bgr, (x, label_y1), (x + text_w + 10, label_y2), color, cv2.FILLED)
        
        # Put text on top
        cv2.putText(
            img_bgr, 
            label, 
            (x + 5, label_y2 - 5), 
            font, 
            font_scale, 
            (0, 0, 0) if name != "Unknown" else (255, 255, 255), 
            font_thickness, 
            cv2.LINE_AA
        )
        
    return img_bgr, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recognize Faces in Image")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output_recognized.jpg', help='Path to save annotated output')
    parser.add_argument('--db', type=str, default='models/embeddings.pkl', help='Path to database')
    parser.add_argument('--method', type=str, default='mtcnn', choices=['mtcnn', 'haar'], help='Detection method')
    parser.add_argument('--threshold', type=float, default=0.60, help='Similarity threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' does not exist.")
        exit(1)
        
    img = cv2.imread(args.image)
    annotated_img, results = recognize_faces_in_image(
        img, 
        database_path=args.db, 
        detection_method=args.method, 
        similarity_threshold=args.threshold
    )
    
    cv2.imwrite(args.output, annotated_img)
    print(f"Successfully processed image. Saved output to {args.output}")
    print("Results:")
    for i, res in enumerate(results):
        print(f"Face {i+1}: Name={res['name']}, Similarity={res['similarity']:.4f}, Box={res['box']}")
