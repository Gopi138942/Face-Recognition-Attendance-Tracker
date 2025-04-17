
### Face Recognition Attendance Tracker

## Description
This project allows you to track the attendance of students in an online class using face recognition. The system matches the face from an uploaded class video with the students' profile pictures and tracks attendance based on their presence for more than 80% of the video duration.
check out here:
https://huggingface.co/spaces/gopi135942/Face_recognition_attendance_Tracker


## Requirements

1. Install the required dependencies:
2. Upload the student profile pictures and class video.
3. The system will process the video, and the attendance will be saved in a CSV file in the `attendance_records/` folder.
4. The attendance records will also be displayed on the Streamlit dashboard.

## Sample Data

- **Profile Pictures:** Upload student images with the filenames as student IDs (e.g., `123.jpg`).
- **Class Video:** Upload a class video where the students' faces are visible for recognition.

## Notes

- The system will automatically mark absent those students who are present for less than 80% of the video duration.
- You can modify the `students_list` in the script to match your actual student list.

  FaceRecognitionAttendance/
├── data/
│   ├── profile_pictures/     # Store student profile images here (e.g., 123.jpg, 456.jpg)
│   ├── attendance_videos/    # Store uploaded class videos here
│   └── attendance_records/   # Attendance CSVs will be saved here
├── app.py                    # Streamlit app file
├── attendance_tracker.py     # Face recognition and attendance tracking script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation


