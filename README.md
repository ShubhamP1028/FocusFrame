# FocusFrame 🎯

An intelligent desktop application built with OpenCV and Python that acts as your personal productivity and posture coach. It uses your webcam to monitor your presence, focus, and sitting habits, helping you stay on task and maintain ergonomic health.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

*   **Smart Session Timer:** Tracks your productive time automatically. The timer pauses when you walk away from your desk and resumes when you return.
*   **Posture Check Alerts:** Gently notifies you if you start slouching or leaning too close to your screen, helping you avoid strain and fatigue.
*   **Focus Mode (Planned):** A planned feature to integrate with system-level tools, temporarily blocking distractions when you are not present at your desk.
*   **Minimalist UI:** A clean, non-intrusive heads-up display (HUD) shows your session stats and status without getting in your way.
*   **Privacy-First:** All processing is done locally on your machine. No video or data is ever sent to the cloud.

## 🛠️ How It Works

FocusFrame uses computer vision algorithms to understand your state at your desk:

1.  **Face Detection:** Utilizes OpenCV's Deep Neural Network (DNN) module with a pre-trained Caffe model to reliably detect your face in real-time.
2.  **Presence Inference:** Tracks the bounding box of your face across video frames. A consistent absence triggers the timer to pause.
3.  **Posture Analysis:** Calculates the size and position of the detected face:
    *   **Distance:** A sudden increase in face size indicates leaning in too close to the screen.
    *   **Position:** A face consistently low in the frame suggests a slouched, hunched posture.
4.  **System Integration:** Uses libraries like `pyautogui` to deliver desktop notifications and alerts.

## 📦 Installation

### Prerequisites
*   Python 3.8 or higher
*   A webcam

### Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/FocusFrame.git
    cd FocusFrame
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  Ensure your webcam is connected and unobstructed.
2.  Run the application from the project directory:
    ```bash
    python main.py
    ```
3.  Sit down at your desk. You should see the video feed with a face detection bounding box.
4.  The timer in the top-left corner will start automatically when you are detected.
5.  A notification will alert you if poor posture is detected.
6.  Minimize the window to keep the application running in the background.

## 📁 Project Structure

FocusFrame/
├── main.py # Main application script

├── models/ # Directory for pre-trained models

│   └── res10_300x300_ssd_iter_140000.caffemodel

│   └── deploy.prototxt

├── requirements.txt # Python dependencies

├── README.md # This file

└── LICENSE # MIT License

---

## 🔮 Future Enhanceances

*   [ ] **Pose Estimation:** Integrate a full body pose model (e.g., MediaPipe) for more accurate and robust posture analysis.
*   [ ] **Focus Mode Integration:** Develop a browser extension or system service to block distracting websites during focused sessions.
*   [ ] **Data Dashboard:** Create a web-based dashboard to visualize your daily focus trends and posture history.
*   [ ] **Cross-Platform Packaging:** Package the app into an executable (.exe, .dmg, .AppImage) for easy installation without requiring Python.

## 🤝 Contributing

Contributions, ideas, and bug reports are welcome! Feel free to fork this project, open an issue, or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more information.

## 🙏 Acknowledgments

*   OpenCV community for the incredible computer vision library.
*   The pre-trained face detection model provided by OpenCV.

---

**Disclaimer:** This project is intended for personal use and learning. It is not a certified medical or ergonomic tool.
