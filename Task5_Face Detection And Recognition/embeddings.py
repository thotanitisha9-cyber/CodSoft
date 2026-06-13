import numpy as np
import torch
from PIL import Image

# Cached embedding model
_EMBEDDING_MODEL = None

def get_embedding_model():
    """Loads and caches the InceptionResnetV1 model pre-trained on VGGFace2."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from facenet_pytorch import InceptionResnetV1
        _EMBEDDING_MODEL = InceptionResnetV1(pretrained='vggface2').eval()
    return _EMBEDDING_MODEL

def preprocess_face(face_img):
    """
    Preprocesses a cropped face image to a normalized PyTorch tensor.
    
    Parameters:
        face_img (np.ndarray or PIL.Image.Image): Cropped RGB face image.
        
    Returns:
        torch.Tensor: Normalized tensor of shape [3, 160, 160]
    """
    if isinstance(face_img, np.ndarray):
        # Ensure it is RGB
        face_pil = Image.fromarray(face_img.astype(np.uint8))
    elif isinstance(face_img, Image.Image):
        face_pil = face_img
    else:
        raise ValueError("face_img must be a numpy array or a PIL Image")

    # FaceNet expects 160x160 images
    face_pil = face_pil.resize((160, 160), Image.Resampling.BILINEAR)
    
    # Convert to float numpy array
    img_data = np.array(face_pil, dtype=np.float32)
    
    # FaceNet standard normalization: (x - 127.5) / 128.0
    img_data = (img_data - 127.5) / 128.0
    
    # Change shape from HWC to CHW
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Convert to tensor
    return torch.tensor(img_data, dtype=torch.float32)

def get_face_embedding(face_img):
    """
    Generates a unit-normalized 512-dimensional embedding for a cropped face image.
    
    Parameters:
        face_img (np.ndarray or PIL.Image.Image): Cropped RGB face image.
        
    Returns:
        np.ndarray: 512-dimensional vector normalized to unit length.
    """
    model = get_embedding_model()
    face_tensor = preprocess_face(face_img)
    
    # Add batch dimension: [1, 3, 160, 160]
    face_tensor = face_tensor.unsqueeze(0)
    
    with torch.no_grad():
        embedding_tensor = model(face_tensor)
        
    embedding = embedding_tensor.squeeze(0).cpu().numpy()
    
    # L2 normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
        
    return embedding

def compute_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two unit-normalized embeddings.
    Since they are L2-normalized, cosine similarity is just the dot product.
    
    Returns:
        float: Cosine similarity score in range [-1, 1]
    """
    return float(np.dot(embedding1, embedding2))

def match_embedding(query_emb, database, threshold=0.60):
    """
    Matches a query embedding against a database of known face embeddings.
    
    Parameters:
        query_emb (np.ndarray): 512-dimensional normalized query embedding.
        database (dict): Dictionary mapping names (str) to lists of normalized embeddings (list of np.ndarray).
        threshold (float): Similarity threshold above which a face is considered recognized.
        
    Returns:
        tuple: (matched_name, similarity_score)
               If no match exceeds the threshold, returns ("Unknown", best_score)
    """
    if not database:
        return "Unknown", 0.0
        
    best_name = "Unknown"
    best_score = -1.0
    
    for name, embeddings in database.items():
        for db_emb in embeddings:
            sim = compute_similarity(query_emb, db_emb)
            if sim > best_score:
                best_score = sim
                if sim >= threshold:
                    best_name = name
                    
    # Double check that our best match actually exceeds threshold
    if best_score < threshold:
        best_name = "Unknown"
        
    return best_name, best_score

if __name__ == "__main__":
    print("Testing Face Embeddings Module...")
    # Create two random face-like image arrays (3 channels, 160x160)
    img1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    try:
        emb1 = get_face_embedding(img1)
        emb2 = get_face_embedding(img2)
        print(f"Successfully generated embeddings of shape: {emb1.shape}")
        
        sim = compute_similarity(emb1, emb2)
        print(f"Cosine similarity between random images: {sim:.4f}")
        
        # Test self-similarity (should be 1.0)
        self_sim = compute_similarity(emb1, emb1)
        print(f"Cosine similarity with itself (should be 1.0): {self_sim:.4f}")
    except Exception as e:
        print(f"Error occurred during embedding generation: {e}")
