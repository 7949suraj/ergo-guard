# 🧙‍♂️ Ergo-Guard: Posture Monitoring App

Ergo-Guard is a posture-detection web app built with **Streamlit**, **OpenCV**, and **MediaPipe**.
It uses your webcam feed to analyze your neck, torso, and proximity angles — giving real-time feedback like:

✅ **Good Posture**
⚠️ **Warning (Slight Lean)**
❌ **Bad Posture**

The system helps users develop healthier sitting habits and can be customized for different thresholds.

---

## 🗂️ Project Structure

```
ergo-guard/
│
├── app.py                   # Main Streamlit app for live posture detection
│
├── evidence/
│   ├── screenshots/             # Contains saved posture screenshots (good/bad/warning)
│   └── DES646 Final Project Report_ Ergo-Guard.docx              # Project documentation report
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/7949suraj/ergo-guard.git
cd ergo-guard
```

### 2️⃣ Create and Activate Virtual Environment

**Windows (PowerShell):**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux (bash):**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

> 💡 If you face any issue with OpenCV headless builds, replace the line in `requirements.txt` with:
>
> ```
> opencv-python==4.8.0.76
> ```

---

## ▶️ Run the App

Run the following command in the project directory:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 🧩 How It Works

1. **Posture Detection:**
   Uses **MediaPipe Pose** to detect body landmarks (neck, torso, shoulders).
2. **Angle Calculation:**
   Calculates the tilt and proximity of upper body joints.
3. **ML Classification:**
   Classifies posture as Good, Warning, or Bad based on trained thresholds.
4. **Real-time Feedback:**
   The Streamlit interface updates continuously with posture label and frame capture.

---

## ⚙️ Customizing Thresholds

You can modify the threshold values in `app.py` under this section:

```python
# Example thresholds
NECK_ANGLE_THRESHOLD = 20
TORSO_ANGLE_THRESHOLD = 15
PROXIMITY_RATIO_THRESHOLD = 0.8
```

Adjust these numbers if your camera angle or sitting distance differs.

---

## 📸 Evidence Folder

* `evidence/screenshots/` — contains saved posture snapshots (e.g., `GoodPosture.png`, `BadPosture.png`)
* `evidence/report.docx` — full project documentation report

---

## 🎥 Live Demonstration Video

🎮 Watch our **1–2 minute project demonstration** here:
👉 [Google Drive Demo Video Link](https://drive.google.com/your-demo-link-here)

The video includes:

* Launching the app via `streamlit run src/app.py`
* Showing posture classifications (Good / Warning / Bad)
* Adjusting thresholds

---

## 👥 Team Members

| Member             | Role                     | Contribution Summary                                                  |
| :----------------- | :----------------------- | :-------------------------------------------------------------------- |
| **Nipun Nandwani(240700)** | Developer & Collaborator | Worked on Streamlit interface, testing, and documentation             |
| **Manas Tripathi( 240616)** | Developer & Collaborator | Helped in model integration, threshold tuning, and report preparation |
| **Suraj Patel(241066)**    | Developer & Collaborator | Assisted in Streamlit UI, documentation, and demo video creation      |

🧩 All members contributed equally to development, testing, documentation, and report preparation.

## 🗓 License

This project is developed for academic / demonstration purposes.
Free to use for learning, testing, or improvement with proper credits.

---

## 💬 Acknowledgements

* [MediaPipe](https://developers.google.com/mediapipe)
* [OpenCV](https://opencv.org)
* [Streamlit](https://streamlit.io)
