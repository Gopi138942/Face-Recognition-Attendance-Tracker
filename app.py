import gradio as gr
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from database import AttendanceDB
from face_utils import FaceRecognizer
import os

# Initialize components
db = AttendanceDB()
recognizer = FaceRecognizer()

def register_student(student_id, name, image):
    """Handle student registration"""
    try:
        # Convert Gradio image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get face embedding
        embedding = recognizer.get_embedding(image)
        if embedding is None:
            return "No face detected - please try another photo"
        
        # Save to database
        db.add_student(student_id, name, embedding)
        return f"Student {name} registered successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

def process_attendance(video_path, class_id, min_attendance=80):
    """Process video for attendance tracking"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize tracking
        students = db.get_students()
        attendance = {s["student_id"]: {"count": 0, "name": s["name"]} for s in students}
        
        # Process every 5th frame (balance accuracy/performance)
        frame_interval = max(1, int(fps / 5))
        
        for frame_num in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect and recognize faces
            faces = recognizer.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                embedding = recognizer.get_embedding(face_img)
                
                if embedding:
                    # Compare with registered students
                    for student in students:
                        similarity = recognizer.compare_embeddings(
                            embedding, 
                            student["embedding"]
                        )
                        if similarity > recognizer.similarity_threshold:
                            attendance[student["student_id"]]["count"] += 1
                            break
        
        cap.release()
        
        # Generate report
        min_frames = (min_attendance/100) * (total_frames/frame_interval)
        report = []
        
        for sid, data in attendance.items():
            duration = (data["count"]/fps) * frame_interval
            status = "Present" if data["count"] >= min_frames else "Absent"
            
            # Record in database
            db.record_attendance(class_id, sid, duration, status)
            
            report.append({
                "Student ID": sid,
                "Name": data["name"],
                "Duration (min)": round(duration/60, 1),
                "Status": status
            })
        
        return pd.DataFrame(report)
    
    except Exception as e:
        return f"Processing error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Attendance Tracker") as app:
    gr.Markdown("# 🎓 Face Recognition Attendance System")
    
    with gr.Tab("Student Registration"):
        gr.Markdown("### Register New Students")
        student_id = gr.Textbox(label="Student ID")
        student_name = gr.Textbox(label="Full Name")
        student_image = gr.Image(label="Face Photo", type="pil")
        register_btn = gr.Button("Register")
        reg_output = gr.Textbox(label="Result")
        
        register_btn.click(
            register_student,
            inputs=[student_id, student_name, student_image],
            outputs=reg_output
        )
    
    with gr.Tab("Class Attendance"):
        gr.Markdown("### Process Class Recording")
        video_input = gr.Video(label="Upload Video")
        class_id = gr.Textbox(label="Class ID")
        min_attendance = gr.Slider(50, 100, 80, label="Minimum Attendance %")
        process_btn = gr.Button("Process")
        attendance_output = gr.Dataframe(label="Attendance Report")
        
        process_btn.click(
            process_attendance,
            inputs=[video_input, class_id, min_attendance],
            outputs=attendance_output
        )

# For Hugging Face deployment
app.launch(debug=True)