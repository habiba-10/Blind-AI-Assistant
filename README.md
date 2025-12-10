# 👁️ Blind AI Assistant (Third Eye)

> **Empowering the visually impaired with an AI-powered "Third Eye" that sees, reads, and speaks.**

## 📌 Project Overview
**Blind AI Assistant** is a web-based assistive technology solution designed to help blind and visually impaired individuals navigate their surroundings independently. 

Unlike standard object detectors, this system provides **Spatial Awareness** (e.g., *"Chair on the left"*), **Smart Voice Feedback**, and a **Mobile-First Tactile Interface**. It combines state-of-the-art Computer Vision models (**YOLOv8x**) for high-accuracy detection with **OCR** for reading texts, all controlled via intuitive **Voice Commands**.

---

## 🌟 Key Features

### 1. 🧠 Intelligent Object Detection
* Powered by **YOLOv8x (Extra Large)** for maximum accuracy.
* **Spatial Awareness:** Doesn't just name objects, but locates them (Left, Right, Ahead).
* **Smart Silence:** Prevents repetitive spamming. It only speaks when the scene changes or new objects appear.

### 2. 📖 OCR (Text Reading Mode)
* Integrated **EasyOCR** engine to read documents, signs, and labels instantly.
* Just say *"Read"* to switch modes.

### 3. 🗣️ Hands-Free Voice Control
* Full control using microphone commands:
    * Say **"Start"** / **"Stop"** to control the camera.
    * Say **"Read"** for Text Mode.
    * Say **"Object"** for Detection Mode.
* **Force Restart:** A dedicated center button to reset the microphone if the browser sleeps.

### 4. 📱 Mobile-First Design
* **Tactile Circle UI:** Designed for thumbs-reach usage, making it easy for visually impaired users to find buttons by touch.
* **High Contrast Dark Mode:** Optimized for battery saving and low-vision users.

---

## 🛠️ Technical Stack

* **Backend:** Python (FastAPI) - Asynchronous & High Performance.
* **AI Models:** Ultralytics YOLOv8x + EasyOCR.
* **Frontend:** HTML5, CSS3, Vanilla JavaScript.
* **Speech:** Web Speech API (Synthesis & Recognition).
* **Deployment Strategy:** Edge Computing (Localhost) exposed via Ngrok for low latency and privacy.

---

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Blind-AI-Assistant.git](https://github.com/YourUsername/Blind-AI-Assistant.git)
    cd Blind-AI-Assistant
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server:**
    ```bash
    uvicorn web_app.main:app --reload
    ```

4.  **Access the App:**
    * Open `http://localhost:8000` in your browser.
    * **Mobile Access:** Use **Ngrok** (`ngrok http 8000`) to get a secure HTTPS link for your phone.

---

## ⚠️ Note on Model Weights
Due to GitHub file size limits, the **`yolov8x.pt`** model is not included in this repo.
* It will download automatically on the first run.
* Or you can download it manually from [Ultralytics](https://github.com/ultralytics/assets/releases) and place it in the root folder.

---

## 🤝 Future Enhancements
* [ ] Currency Recognition Model (EGP/USD).
* [ ] Face Recognition (Identify friends/family).
* [ ] Distance Estimation (using depth estimation models).

---

### 📬 Contact
Created by **[Your Name]** - Feel free to contact me!