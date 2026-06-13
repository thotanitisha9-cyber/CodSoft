import os
import time
import pickle
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detect_faces import detect_faces
from embeddings import get_face_embedding, match_embedding
from train import train_system, evaluate_system
from recognize_faces import recognize_faces_in_image, load_embeddings_database, clear_database_cache

# Set page layout
st.set_page_config(
    page_title="AI Face Detection & Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom Headers */
.main-title {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #a3b3c6;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Metric Cards Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #1a1e28;
    border: 1px solid #2d3345;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 242, 254, 0.2);
    border-color: #00f2fe;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #00f2fe;
}

.metric-label {
    font-size: 0.85rem;
    color: #8c9ba5;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Instructions card */
.info-card {
    background-color: #121620;
    border-left: 5px solid #4facfe;
    border-radius: 4px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}

.info-title {
    font-weight: 600;
    color: #f0f2f6;
    margin-bottom: 0.5rem;
}

.info-text {
    color: #a3b3c6;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# Helper function to check if models and database are ready
def get_db_status():
    db_path = 'models/embeddings.pkl'
    if not os.path.exists(db_path):
        return False, {}
    try:
        db = load_embeddings_database(db_path)
        return len(db) > 0, db
    except:
        return False, {}

# Top banner
st.markdown('<div class="main-title">Face Detection & Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep learning-powered biometric vision pipeline with real-time analytics</div>', unsafe_allow_html=True)

# Navigation
tabs = st.tabs([
    "📸 Image Recognition", 
    "🎥 Video Recognition", 
    "💻 Live Webcam Stream", 
    "🗂️ Dataset & Model Manager"
])

# Sidebar configuration
st.sidebar.markdown("### ⚙️ Pipeline Settings")
detection_method = st.sidebar.selectbox(
    "Face Detection Backend",
    options=["MTCNN", "Haar Cascade"],
    index=0,
    help="MTCNN is deep learning-based and more accurate. Haar Cascade is faster on older CPUs."
)

similarity_threshold = st.sidebar.slider(
    "Recognition Similarity Threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.60,
    step=0.05,
    help="Higher threshold means stricter matching (fewer False Acceptances, more Unknowns)."
)

# Convert display name to backend code
det_method_code = 'mtcnn' if detection_method == "MTCNN" else 'haar'

# ----------------- Tab 1: Image Recognition -----------------
with tabs[0]:
    st.markdown("### Process Static Images")
    
    is_trained, db = get_db_status()
    if not is_trained:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">⚠️ Model Database Empty</div>
            <div class="info-text">
                No face embeddings were found in the database. 
                Please head to the <b>Dataset & Model Manager</b> tab to add photos and train the system first!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg", "webp"])
    
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(img_rgb, use_container_width=True)
            
        with col2:
            st.markdown("**Recognition Output**")
            
            # Run recognition
            with st.spinner("Processing face detection and embedding comparison..."):
                annotated_bgr, results = recognize_faces_in_image(
                    img_bgr,
                    database_path='models/embeddings.pkl',
                    detection_method=det_method_code,
                    similarity_threshold=similarity_threshold
                )
                
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
            
            # Save output for download
            is_success, buffer = cv2.imencode('.jpg', annotated_bgr)
            if is_success:
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buffer.tobytes(),
                    file_name="annotated_output.jpg",
                    mime="image/jpeg"
                )
                
        # Show tabular results
        if results:
            st.markdown("### Detection Metadata")
            res_df = []
            for i, res in enumerate(results):
                res_df.append({
                    "Face Index": i + 1,
                    "Recognized Person": res["name"],
                    "Confidence / Similarity Score": f"{res['similarity']:.2%}" if res["name"] != "Unknown" else "N/A",
                    "Detection Confidence": f"{res['detection_confidence']:.2%}",
                    "Bounding Box [x, y, w, h]": str(res["box"])
                })
            st.dataframe(pd.DataFrame(res_df), use_container_width=True)
        else:
            st.info("No faces detected in the uploaded image.")

# ----------------- Tab 2: Video Recognition -----------------
with tabs[1]:
    st.markdown("### Process Video Files")
    
    is_trained, db = get_db_status()
    if not is_trained:
        st.info("Please train the system in the 'Dataset & Model Manager' tab before processing video.")
        
    uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None and is_trained:
        # Save uploaded video to temp file
        temp_input_path = "temp_input_video.mp4"
        temp_output_path = "temp_output_video.mp4"
        
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())
            
        st.video(temp_input_path)
        
        if st.button("🚀 Start Video Face Recognition"):
            cap = cv2.VideoCapture(temp_input_path)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                total_frames = 100 # safety fallback
                
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps_val, (width, height))
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            frame_idx = 0
            start_time = time.time()
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Process frame
                    annotated_frame, _ = recognize_faces_in_image(
                        frame,
                        database_path='models/embeddings.pkl',
                        detection_method=det_method_code,
                        similarity_threshold=similarity_threshold
                    )
                    
                    out.write(annotated_frame)
                    frame_idx += 1
                    
                    # Update progress
                    progress = min(1.0, frame_idx / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_idx}/{total_frames}... ({progress:.1%})")
                    
                cap.release()
                out.release()
                
                processing_time = time.time() - start_time
                status_text.success(f"Video processing completed! Processed {frame_idx} frames in {processing_time:.1f}s.")
                
                # Download button for output video
                with open(temp_output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=f.read(),
                        file_name="annotated_video.mp4",
                        mime="video/mp4"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
                if cap.isOpened():
                    cap.release()
                
            # Cleanup temp input
            if os.path.exists(temp_input_path):
                try:
                    os.remove(temp_input_path)
                except:
                    pass

# ----------------- Tab 3: Live Webcam Stream -----------------
with tabs[2]:
    st.markdown("### Real-time Face Recognition via Local Camera")
    
    is_trained, db = get_db_status()
    if not is_trained:
        st.info("Please train the system in the 'Dataset & Model Manager' tab before launching live stream.")
        
    st.markdown("""
    When running on your local machine, this feature links directly with your webcam. 
    Click the toggle below to activate the video stream. Bounding boxes and recognition overlays are rendered frame-by-frame.
    """)
    
    run_webcam = st.toggle("🔌 Turn On Camera Feed", key="webcam_toggle_switch")
    
    if run_webcam and is_trained:
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not access camera device. Please ensure your webcam is connected and not in use by another app.")
        else:
            # Placeholders for Streamlit display
            st_frame = st.empty()
            st_info = st.empty()
            
            prev_time = time.time()
            frame_count = 0
            fps = 0.0
            
            try:
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera.")
                        break
                        
                    # Calculate FPS
                    curr_time = time.time()
                    frame_count += 1
                    if curr_time - prev_time >= 1.0:
                        fps = frame_count / (curr_time - prev_time)
                        frame_count = 0
                        prev_time = curr_time
                        
                    # Recognize faces
                    annotated_frame, results = recognize_faces_in_image(
                        frame,
                        database_path='models/embeddings.pkl',
                        detection_method=det_method_code,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Draw FPS overlay
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
                    
                    # Display the frame in Streamlit
                    # Convert BGR to RGB for Streamlit display
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(annotated_rgb, channels="RGB", use_container_width=True)
                    
                    # Log active faces detected
                    active_faces = [r["name"] for r in results]
                    detected_str = ", ".join(active_faces) if active_faces else "None"
                    st_info.markdown(f"**Webcam Status:** Active | **FPS:** {fps:.1f} | **Active Detections:** {detected_str}")
                    
                    # Yield control to prevent blocking Streamlit rerun
                    time.sleep(0.01)
                    
            finally:
                cap.release()
                st_frame.empty()
                st_info.info("Camera feed stopped.")

# ----------------- Tab 4: Dataset & Model Manager -----------------
with tabs[3]:
    st.markdown("### Manage Face Dataset & Train System")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("#### ➕ Add New Person to Database")
        with st.form("new_person_form", clear_on_submit=True):
            person_name = st.text_input("Person Name (e.g. John_Doe)", placeholder="Enter Name").strip()
            uploaded_faces = st.file_uploader(
                "Upload Face Images (Provide at least 3-5 clear shots of different angles/expressions)", 
                type=["jpg", "png", "jpeg", "webp"],
                accept_multiple_files=True
            )
            
            submit_btn = st.form_submit_button("📁 Add Images to Dataset")
            
            if submit_btn:
                if not person_name:
                    st.error("Please enter a name for the person.")
                elif not uploaded_faces:
                    st.error("Please upload at least one image.")
                else:
                    # Sanitize folder name
                    safe_name = "".join(x for x in person_name if (x.isalnum() or x in "._- ")).replace(" ", "_")
                    person_dir = os.path.join("dataset", safe_name)
                    os.makedirs(person_dir, exist_ok=True)
                    
                    saved_count = 0
                    for uploaded_file in uploaded_faces:
                        try:
                            # Save file
                            file_path = os.path.join(person_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.read())
                            saved_count += 1
                        except Exception as e:
                            st.error(f"Error saving {uploaded_file.name}: {e}")
                            
                    st.success(f"Successfully saved {saved_count} images for **{safe_name}** in the dataset folder!")
                    clear_database_cache()
                    
        # View dataset status
        st.markdown("#### 📂 Current Dataset Inventory")
        if os.path.exists("dataset"):
            dirs = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]
            if not dirs:
                st.info("The dataset directory is empty. Add a new person above.")
            else:
                inventory = []
                for d in dirs:
                    path = os.path.join("dataset", d)
                    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
                    inventory.append({"Person Name": d, "Image Count": len(files)})
                st.dataframe(pd.DataFrame(inventory), use_container_width=True)
        else:
            st.info("No dataset folder found. It will be created when you add a person.")
            
    with col_r:
        st.markdown("#### 🦾 Model Training & Cross-Validation")
        st.markdown("""
        Click the button below to parse all images in the `dataset/` directory, compute 512-D embeddings using InceptionResnetV1, 
        and compile the database in `models/embeddings.pkl`.
        
        The training pipeline will also run a **Leave-One-Out validation** to measure real biometric classification metrics.
        """)
        
        # Pull saved metrics from file or session state if trained
        if 'eval_metrics' not in st.session_state:
            st.session_state['eval_metrics'] = None
            
        train_btn = st.button("⚙️ Train Face Recognition Model")
        
        if train_btn:
            with st.spinner("Extracting face features and optimizing embeddings database..."):
                clear_database_cache()
                database, metrics = train_system(
                    dataset_dir='dataset',
                    models_dir='models',
                    detection_method=det_method_code,
                    similarity_threshold=similarity_threshold
                )
                
                if database:
                    st.success("Model trained successfully! Saved to models/embeddings.pkl.")
                    st.session_state['eval_metrics'] = metrics
                else:
                    st.error("Training failed. Ensure your dataset directory contains subfolders with face images.")
                    
        # Display metrics using custom cards if available
        metrics = st.session_state['eval_metrics']
        if metrics is not None:
            st.markdown("#### 📊 Evaluation Analytics")
            st.markdown(f"Metrics computed over **{metrics['total_samples']} samples** using Leave-One-Out cross-validation:")
            
            # HTML for styled card components
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{metrics['accuracy']:.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['precision_macro']:.1%}</div>
                    <div class="metric-label">Macro Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['recall_macro']:.1%}</div>
                    <div class="metric-label">Macro Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['f1_macro']:.1%}</div>
                    <div class="metric-label">Macro F1 Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            > [!TIP]
            > **Macro average** measures the metric independently for each person and averages them. 
            > This guarantees that performance on minority classes is weighed equally with majority classes, 
            > showing true biometric performance across all registered individuals.
            """)
        elif is_trained:
            st.info("Click 'Train Face Recognition Model' above to evaluate system performance and display metrics.")
