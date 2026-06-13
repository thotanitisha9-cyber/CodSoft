import time
import cv2
import argparse
from recognize_faces import recognize_faces_in_image

def start_webcam_recognition(database_path='models/embeddings.pkl', detection_method='mtcnn', similarity_threshold=0.60):
    """
    Starts a real-time OpenCV window that captures webcam feed and runs face recognition.
    """
    print("Initializing webcam capture. Press 'q' to exit.")
    
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    prev_time = time.time()
    
    # Create window and set properties
    window_name = "Face Detection and Recognition System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Compute FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time
            
            # Process frame (image is in BGR format)
            annotated_frame, results = recognize_faces_in_image(
                frame,
                database_path=database_path,
                detection_method=detection_method,
                similarity_threshold=similarity_threshold
            )
            
            # Draw FPS overlay
            # Use a modern color (Cyan - BGR: 255, 255, 0)
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.1f}", 
                (15, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.8, 
                (255, 255, 0), 
                2, 
                cv2.LINE_AA
            )
            
            # Draw instruction overlay
            cv2.putText(
                annotated_frame, 
                "Press 'q' to Quit", 
                (15, annotated_frame.shape[0] - 15), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.6, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Check if window was closed via close 'X' button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam capture stopped cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Face Recognition via Webcam")
    parser.add_argument('--db', type=str, default='models/embeddings.pkl', help='Path to database')
    parser.add_argument('--method', type=str, default='mtcnn', choices=['mtcnn', 'haar'], help='Detection method')
    parser.add_argument('--threshold', type=float, default=0.60, help='Similarity threshold')
    
    args = parser.parse_args()
    start_webcam_recognition(
        database_path=args.db,
        detection_method=args.method,
        similarity_threshold=args.threshold
    )
