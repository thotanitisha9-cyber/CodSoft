import os
import pickle
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from detect_faces import detect_faces
from embeddings import get_face_embedding, match_embedding

def train_system(dataset_dir='dataset', models_dir='models', detection_method='mtcnn', similarity_threshold=0.60):
    """
    Scans the dataset directory, extracts face embeddings for each person,
    saves the database, and evaluates performance.
    """
    print(f"Starting Training Phase using detection method: {detection_method}...")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created empty dataset directory at '{dataset_dir}'. Please add person directories and face images.")
        return None, None
        
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    database = {}
    total_images_processed = 0
    faces_detected_count = 0
    failed_images = []
    
    # Iterate through each folder (person) in dataset/
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue
            
        print(f"Processing images for: {person_name}")
        database[person_name] = []
        
        # Iterate through images in the person's folder
        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                continue
                
            total_images_processed += 1
            img_path = os.path.join(person_path, img_file)
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  [Error] Could not read image: {img_file}")
                    failed_images.append(img_path)
                    continue
                    
                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                detections = detect_faces(img_rgb, method=detection_method)
                
                if not detections:
                    # Fallback: if no face is detected, we can try Haar Cascade as fallback
                    if detection_method == 'mtcnn':
                        detections = detect_faces(img_rgb, method='haar')
                        
                if not detections:
                    # Final fallback: use the entire image as the face crop (for pre-cropped face datasets)
                    print(f"  [Warning] No face detected in {img_file}. Using entire image as crop fallback.")
                    face_crop = img_rgb
                else:
                    # Use the face crop from the best detection (largest box or highest confidence)
                    # We sort by box area or confidence. Let's just take the first detection.
                    face_crop = detections[0]['face_img']
                    faces_detected_count += 1
                    
                # Extract face embedding
                embedding = get_face_embedding(face_crop)
                database[person_name].append(embedding)
                
            except Exception as e:
                print(f"  [Error] Failed to process {img_file}: {e}")
                failed_images.append(img_path)
                
        # If no embeddings were successfully extracted, remove the person key
        if not database[person_name]:
            del database[person_name]
            print(f"  [Warning] No valid face embeddings generated for {person_name}.")
            
    # Save the database
    embeddings_file = os.path.join(models_dir, 'embeddings.pkl')
    with open(embeddings_file, 'wb') as f:
        pickle.dump(database, f)
        
    print("\n--- Training Phase Summary ---")
    print(f"Total images processed: {total_images_processed}")
    print(f"Faces successfully detected: {faces_detected_count}")
    print(f"Embeddings saved to: {embeddings_file}")
    print(f"Classes in database: {list(database.keys())}")
    
    if failed_images:
        print(f"Failed images ({len(failed_images)}): {failed_images}")
        
    # Run Evaluation
    metrics = evaluate_system(database, similarity_threshold)
    return database, metrics

def evaluate_system(database, threshold=0.60):
    """
    Evaluates the face recognition system using Leave-One-Out cross-validation.
    """
    print("\nEvaluating system performance...")
    
    # Flatten database
    X = []
    y_true = []
    
    for name, embs in database.items():
        for emb in embs:
            X.append(emb)
            y_true.append(name)
            
    total_samples = len(X)
    if total_samples < 2:
        print("Evaluation skipped: Need at least 2 images across the dataset to evaluate.")
        return None
        
    y_pred = []
    
    for i in range(total_samples):
        # Build temporary database without the i-th embedding
        temp_database = {}
        for j in range(total_samples):
            if i == j:
                continue
            name_j = y_true[j]
            emb_j = X[j]
            if name_j not in temp_database:
                temp_database[name_j] = []
            temp_database[name_j].append(emb_j)
            
        # Match the query embedding against the rest of the database
        pred_name, score = match_embedding(X[i], temp_database, threshold=threshold)
        y_pred.append(pred_name)
        
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Micro/macro metrics
    # Note: predicted labels can include "Unknown" (not in y_true)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'total_samples': total_samples
    }
    
    print("--- Evaluation Metrics ---")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1 Score:  {f1_macro:.4f}")
    print(f"Micro F1 Score:  {f1_micro:.4f}")
    print(f"Total Eval Samples: {total_samples}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Face Recognition System")
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--models', type=str, default='models', help='Path to models directory')
    parser.add_argument('--method', type=str, default='mtcnn', choices=['mtcnn', 'haar'], help='Detection method')
    parser.add_argument('--threshold', type=float, default=0.60, help='Similarity threshold')
    
    args = parser.parse_args()
    train_system(
        dataset_dir=args.dataset,
        models_dir=args.models,
        detection_method=args.method,
        similarity_threshold=args.threshold
    )
