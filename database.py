import sqlite3
import numpy as np
import pickle
from datetime import datetime

class AttendanceDB:
    def __init__(self, db_path="attendance.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False))
        self.create_tables()
    
    def create_tables(self):
        """Initialize database tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT,
                embedding BLOB
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                class_id TEXT,
                student_id TEXT,
                date TEXT,
                duration REAL,
                status TEXT,
                PRIMARY KEY (class_id, student_id, date)
            )
        """)
        self.conn.commit()
    
    def add_student(self, student_id, name, embedding):
        """Register new student with face embedding"""
        self.conn.execute(
            "INSERT INTO students VALUES (?, ?, ?)",
            (student_id, name, pickle.dumps(embedding)))
        self.conn.commit()
    
    def get_students(self):
        """Retrieve all registered students"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT student_id, name, embedding FROM students")
        return [
            {
                "student_id": row[0],
                "name": row[1],
                "embedding": pickle.loads(row[2])
            } for row in cursor.fetchall()
        ]
    
    def record_attendance(self, class_id, student_id, duration, status):
        """Save attendance record"""
        self.conn.execute(
            "INSERT OR REPLACE INTO attendance VALUES (?, ?, ?, ?, ?)",
            (class_id, student_id, datetime.now().date(), duration, status)
        )
        self.conn.commit()

import threading

def handle_request():
    conn = sqlite3.connect("attendance.db")  # Create a new connection for each thread
    db = AttendanceDB(conn)
    # Your database operations go here
