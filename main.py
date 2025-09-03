import cv2
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import customtkinter as ctk
from PIL import Image, ImageTk
import queue

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class GazeFocus:
    def __init__(self):
        # Core state
        self.is_present = False
        self.posture_multiplier = 1.0
        self.focus_score = 0.0
        self.max_score = 100.0
        
        # Session tracking
        self.session_start_time = None
        self.session_active = False
        self.total_session_time = 0
        self.last_update_time = time.time()
        
        # Posture baseline (calibrated when first face detected)
        self.baseline_face_size = None
        self.baseline_face_center_y = None
        self.calibration_frames = 0
        self.calibration_complete = False
        
        # Scoring parameters
        self.presence_points_per_second = 0.5  # Base points for being present
        self.max_posture_multiplier = 1.5      # Good posture bonus
        self.min_posture_multiplier = 0.3      # Poor posture penalty
        self.score_decay_rate = 0.1            # Points lost per second when away
        
        # Detection parameters
        self.face_size_tolerance = 0.4         # Threshold for "too close" detection
        self.face_position_tolerance = 0.3     # Threshold for slouching detection
        self.detection_smoothing = 3           # Frames to smooth detection
        self.detection_buffer = []
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.opencv_thread = None
        
        # OpenCV setup
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Setup UI
        self.setup_ui()
        self.start_opencv_thread()
    
    def setup_ui(self):
        """Setup modern dark-themed UI"""
        self.root = ctk.CTk()
        self.root.title("GazeFocus - Intelligent Productivity Coach")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        
        # Main container
        main_frame = ctk.CTkFrame(self.root, corner_radius=0)
        main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="GazeFocus",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=(30, 20))
        
        # Video feed frame
        self.video_frame = ctk.CTkFrame(main_frame, width=320, height=240, corner_radius=15)
        self.video_frame.pack(pady=(0, 30))
        self.video_frame.pack_propagate(False)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Camera Loading...", width=320, height=240)
        self.video_label.pack(expand=True)
        
        # Focus Score Section
        score_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        score_frame.pack(fill="x", padx=40, pady=(0, 20))
        
        score_title = ctk.CTkLabel(
            score_frame,
            text="Focus Score",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        score_title.pack(pady=(15, 5))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            score_frame,
            width=400,
            height=20,
            corner_radius=10
        )
        self.progress_bar.pack(pady=(5, 10))
        self.progress_bar.set(0)
        
        # Score percentage
        self.score_label = ctk.CTkLabel(
            score_frame,
            text="100/100",  # Start at 100
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.score_label.pack(pady=(0, 15))
        
        # Status Section
        status_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        status_frame.pack(fill="x", padx=40, pady=(0, 20))
        
        # Status indicators
        status_container = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_container.pack(pady=20)
        
        # Create status rows
        self.status_label = ctk.CTkLabel(
            status_container,
            text="🔴 Not Active",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#ff6b6b"
        )
        self.status_label.pack(pady=5)
        
        self.posture_label = ctk.CTkLabel(
            status_container,
            text="📏 Calibrating...",
            font=ctk.CTkFont(size=16),
            text_color="#ffd93d"
        )
        self.posture_label.pack(pady=5)
        
        self.session_label = ctk.CTkLabel(
            status_container,
            text="🕒 00:00:00",
            font=ctk.CTkFont(size=16),
            text_color="#74c0fc"
        )
        self.session_label.pack(pady=5)
        
        # Control buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=30)
        
        self.start_button = ctk.CTkButton(
            button_frame,
            text="Start Session",
            font=ctk.CTkFont(size=16, weight="bold"),
            width=120,
            height=40,
            corner_radius=20,
            command=self.toggle_session
        )
        self.start_button.pack(side="left", padx=10)
        
        self.reset_button = ctk.CTkButton(
            button_frame,
            text="Reset",
            font=ctk.CTkFont(size=16),
            width=100,
            height=40,
            corner_radius=20,
            fg_color="#ff6b6b",
            hover_color="#ff5252",
            command=self.reset_session
        )
        self.reset_button.pack(side="left", padx=10)
        
        self.quit_button = ctk.CTkButton(
            button_frame,
            text="Quit",
            font=ctk.CTkFont(size=16),
            width=100,
            height=40,
            corner_radius=20,
            fg_color="#6c757d",
            hover_color="#5a6268",
            command=self.quit_app
        )
        self.quit_button.pack(side="left", padx=10)
        
        # Start UI update loop
        self.update_ui()
    
    def start_opencv_thread(self):
        """Start the OpenCV processing thread"""
        self.opencv_thread = threading.Thread(target=self.opencv_loop, daemon=True)
        self.opencv_thread.start()
    
    def opencv_loop(self):
        """Main OpenCV processing loop (runs in separate thread)"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Update scoring
            self.update_focus_score()
            
            # Send frame to UI thread (non-blocking)
            try:
                self.frame_queue.put(processed_frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full
            
            time.sleep(1/30)  # ~30 FPS
    
    def process_frame(self, frame):
        """Process frame for face detection and posture analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        # Smooth detection using buffer
        face_detected = len(faces) > 0
        self.detection_buffer.append(face_detected)
        if len(self.detection_buffer) > self.detection_smoothing:
            self.detection_buffer.pop(0)
        
        # Determine presence based on smoothed detection
        self.is_present = sum(self.detection_buffer) > len(self.detection_buffer) // 2
        
        if face_detected:
            # Get largest face
            face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = face
            
            # Draw face rectangle
            color = (0, 255, 0) if self.is_present else (255, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Calibrate baseline if needed
            if not self.calibration_complete:
                self.calibrate_posture(x, y, w, h)
            
            # Analyze posture
            if self.calibration_complete:
                self.analyze_posture(x, y, w, h)
        else:
            # No face detected
            self.posture_multiplier = 1.0  # Neutral when away
        
        # Add overlay information
        self.add_frame_overlay(frame)
        
        return frame
    
    def calibrate_posture(self, x, y, w, h):
        """Calibrate baseline posture measurements"""
        face_size = w * h
        face_center_y = y + h // 2
        
        if self.baseline_face_size is None:
            self.baseline_face_size = face_size
            self.baseline_face_center_y = face_center_y
            self.calibration_frames = 1
        else:
            # Average over multiple frames for stability
            self.baseline_face_size = (self.baseline_face_size * self.calibration_frames + face_size) / (self.calibration_frames + 1)
            self.baseline_face_center_y = (self.baseline_face_center_y * self.calibration_frames + face_center_y) / (self.calibration_frames + 1)
            self.calibration_frames += 1
        
        if self.calibration_frames >= 30:  # Calibrate over 30 frames (~1 second)
            self.calibration_complete = True
    
    def analyze_posture(self, x, y, w, h):
        """Analyze current posture and update multiplier"""
        face_size = w * h
        face_center_y = y + h // 2
        
        # Calculate deviations from baseline
        size_ratio = face_size / self.baseline_face_size
        y_deviation = abs(face_center_y - self.baseline_face_center_y) / self.baseline_face_center_y
        
        # Determine posture quality
        posture_score = 1.0
        
        # Penalty for being too close (face too large)
        if size_ratio > (1 + self.face_size_tolerance):
            posture_score *= max(0.3, 1.0 - (size_ratio - 1.0))
        
        # Penalty for slouching (face position changed significantly)
        if y_deviation > self.face_position_tolerance:
            posture_score *= max(0.3, 1.0 - y_deviation)
        
        # Bonus for good posture
        if size_ratio <= (1 + self.face_size_tolerance * 0.5) and y_deviation <= self.face_position_tolerance * 0.5:
            posture_score = min(self.max_posture_multiplier, posture_score * 1.2)
        
        self.posture_multiplier = max(self.min_posture_multiplier, min(self.max_posture_multiplier, posture_score))
    
    def add_frame_overlay(self, frame):
        """Add informational overlay to frame"""
        height, width = frame.shape[:2]
        
        # Status indicator
        status_text = "PRESENT" if self.is_present else "AWAY"
        status_color = (0, 255, 0) if self.is_present else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Posture indicator
        if self.calibration_complete:
            if self.posture_multiplier > 1.2:
                posture_text = "EXCELLENT POSTURE"
                posture_color = (0, 255, 0)
            elif self.posture_multiplier > 0.8:
                posture_text = "GOOD POSTURE"
                posture_color = (0, 255, 255)
            else:
                posture_text = "CHECK POSTURE"
                posture_color = (0, 165, 255)
            
            cv2.putText(frame, posture_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
        
        # Focus score
        cv2.putText(frame, f"Score: {int(self.focus_score)}", (10, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def update_focus_score(self):
        """Update focus score based on presence and posture"""
        if not self.session_active:
            return
        
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if self.is_present:
            # Earn points for being present, modified by posture
            points_earned = self.presence_points_per_second * time_delta * self.posture_multiplier
            self.focus_score = min(self.max_score, self.focus_score + points_earned)
        else:
            # Lose points when away
            points_lost = self.score_decay_rate * time_delta
            self.focus_score = max(0, self.focus_score - points_lost)
    
    def toggle_session(self):
        """Toggle session start/pause"""
        if not self.session_active:
            self.start_session()
        else:
            self.pause_session()
    
    def start_session(self):
        """Start a new session"""
        self.session_active = True
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.start_button.configure(text="Pause Session")
    
    def pause_session(self):
        """Pause current session"""
        if self.session_active and self.session_start_time:
            self.total_session_time += time.time() - self.session_start_time
        self.session_active = False
        self.session_start_time = None
        self.start_button.configure(text="Resume Session")
    
    def reset_session(self):
        """Reset all session data"""
        self.pause_session()
        self.focus_score = 0.0
        self.total_session_time = 0
        self.baseline_face_size = None
        self.baseline_face_center_y = None
        self.calibration_frames = 0
        self.calibration_complete = False
        self.posture_multiplier = 1.0
        self.start_button.configure(text="Start Session", fg_color="#1f538d", hover_color="#14375e")
    
    def update_ui(self):
        """Update UI elements (runs in main thread)"""
        if not self.running:
            return
        
        # Update video feed
        try:
            frame = self.frame_queue.get_nowait()
            # Convert frame to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (320, 240))
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
        except queue.Empty:
            pass
        
        # Update focus score
        score_percentage = self.focus_score / self.max_score
        self.progress_bar.set(score_percentage)
        self.score_label.configure(text=f"{int(self.focus_score)}%")
        
        # Update status indicators
        if self.session_active:
            if self.is_present:
                self.status_label.configure(text="🟢 Focused", text_color="#51cf66")
            else:
                self.status_label.configure(text="🟡 Away", text_color="#ffd93d")
        else:
            self.status_label.configure(text="🔴 Paused", text_color="#ff6b6b")
        
        # Update posture status
        if not self.calibration_complete:
            self.posture_label.configure(text="📏 Calibrating...", text_color="#ffd93d")
        elif self.posture_multiplier > 1.2:
            self.posture_label.configure(text="✅ Excellent Posture", text_color="#51cf66")
        elif self.posture_multiplier > 0.8:
            self.posture_label.configure(text="✅ Good Posture", text_color="#51cf66")
        else:
            self.posture_label.configure(text="⚠️ Check Posture", text_color="#ff8787")
        
        # Update session time
        current_session_time = self.total_session_time
        if self.session_active and self.session_start_time:
            current_session_time += time.time() - self.session_start_time
        
        hours = int(current_session_time // 3600)
        minutes = int((current_session_time % 3600) // 60)
        seconds = int(current_session_time % 60)
        self.session_label.configure(text=f"🕒 {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Schedule next update
        self.root.after(100, self.update_ui)  # Update every 100ms
    
    def quit_app(self):
        """Clean shutdown"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.opencv_thread:
            self.opencv_thread.join(timeout=1)
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.mainloop()

if __name__ == "__main__":
    print("Starting FocusFrame - Intelligent Productivity Coach")
    print("\nFeatures:")
    print("✓ Smart presence detection with smoothing")
    print("✓ Real-time posture analysis and scoring")
    print("✓ Focus score with posture multipliers")
    print("✓ Modern dark-themed UI")
    print("✓ Automatic calibration system")
    print("✓ Thread-safe video processing")
    print("\nEnjoy your focused work session!")
    
    try:
        app = GazeFocus()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required packages installed:")
        print("pip install opencv-python customtkinter pillow numpy")