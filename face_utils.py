import cv2
import numpy as np
from deepface import DeepFace  # Uses OpenCV's DNN by default for CPU

class FaceRecognizer:
    def __init__(self):
        # Configuration
        self.detector_backend = "opencv"  # CPU optimized
        self.recognition_model = "ArcFace"  # Best accuracy
        self.similarity_threshold = 0.6  # Minimum match confidence
    
    def detect_faces(self, image):
        """Detect faces using OpenCV DNN"""
        try:
            # Convert to RGB (required by DeepFace)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces with OpenCV backend for CPU efficiency
            faces = DeepFace.extract_faces(
                rgb_image,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            return [face["facial_area"] for face in faces if face["confidence"] > 0.9]
        except:
            return []

    def get_embedding(self, face_image):
        """Extract face embedding using ArcFace"""
        try:
            # Resize and normalize image
            resized = cv2.resize(face_image, (112, 112))
            normalized = resized.astype(np.float32) / 255.0
            
            # Get embedding (ArcFace is more accurate than Facenet on CPU)
            embedding = DeepFace.represent(
                normalized,
                model_name=self.recognition_model,
                enforce_detection=False,
                detector_backend="skip"
            )
            return embedding
        except:
            return None

    def compare_embeddings(self, emb1, emb2):
        """Calculate cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))