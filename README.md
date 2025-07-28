
# 🛡️ Real-Time Helmet Detection Using YOLOv3 and ESP32-CAM

This project is a real-time safety monitoring system that detects helmet usage using the ESP32-CAM module and YOLOv3-Tiny deep learning model. When a rider is detected without a helmet, the system triggers an SMS alert using the Twilio API. Designed for traffic surveillance and industrial safety, it offers cost-effective and scalable enforcement of helmet compliance.

---

## 📂 Project Structure

```
Helmet-Detection/
├── Arduino_Code/              # ESP32-CAM firmware (Arduino IDE)
├── Python_Scripts/            # Python scripts for detection & alerts
├── YOLO_Files/                # YOLOv3 config and .names
├── Model/                     # Trained .h5 CNN model (linked externally)
├── Images/                    # Output screenshots & hardware photos
└── README.md                  # Project documentation
```

---

## 🔧 Technologies Used

- **ESP32-CAM**
- **YOLOv3-Tiny**
- **OpenCV + TensorFlow**
- **Twilio SMS API**
- **Python 3.10, Arduino IDE**
- **IoT & Embedded Systems**

---

## ⚙️ Features

- Real-time helmet detection with camera feed
- YOLOv3-Tiny deep learning model for object detection
- SMS alerts for no-helmet violations using Twilio
- Custom CNN for helmet classification
- Lightweight design for IoT/edge devices

---

## 📥 Download Large Files

> Due to GitHub's file size limits, please download the following files manually:

- 📦 **YOLOv3 Weights** (`yolov3-custom.weights`)  
  👉 [Download from Google Drive](https://drive.google.com/file/d/1I3br6Ih83ATBshdZFcP1cBkZjEj0Ol6U/view?usp=drive_link)

- 🧠 **Trained CNN Model** (`helmet-nonhelmet_cnn.h5`)  
  👉 [Download from Google Drive](https://drive.google.com/file/d/1COHGucmS21SA0EvPnPBTl613lZeJsSxi/view?usp=drive_link)

> After downloading, place the files in the corresponding folders (`YOLO_Files/`, `Model/`, etc.)

---

## 🚀 How to Run

### 1. **ESP32-CAM Setup**
- Flash Arduino code from `Arduino_Code/`
- Configure Wi-Fi credentials
- Use FTDI programmer and follow flashing steps

### 2. **Python Backend**
- Install dependencies:  
  ```bash
  pip install opencv-python tensorflow twilio pytesseract imutils
  ```
- Run:  
  ```bash
  python helmet_detection.py
  ```
---

## 📚 License

This project is for **academic and research purposes only**.  
YOLOv3 is licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).  
Other libraries follow their respective open-source licenses.

---

## 👨‍💻 Contributors
0
- Dorababu(https://github.com/Dorababu70)

---
