# 🚉 RailGuard-AI

**RailGuard-AI** is an AI-powered surveillance and alert system designed to enhance **crowd management**, **waste monitoring**, and **violence detection** at railway stations using CCTV footage and real-time object detection.

---

## 🔍 Overview

RailGuard-AI aims to:

- Improve passenger safety and security
- Automate trash and waste detection
- Detect violent behavior and send real-time alerts
- Optimize operational workflows in railway environments

---

## ⚙️ Tech Stack

### Backend
- **Python**
- **FastAPI / Flask**
- **YOLOv8 / YOLOv8s / YOLO-NAS**
- **OpenCV, PyTorch, NumPy**

### Frontend (Optional Dashboard)
- **React.js**
- **Tailwind CSS / Material UI**
- **WebSockets** (for real-time alerts)

### Data & Deployment
- **Datasets**: COCO, RWF-2000, Roboflow
- **Deployment**: Render / Railway / Local Server
- **Database**: Firebase / PostgreSQL (for logs and events)

---

## ✨ Features

- 🧍‍ Crowd Detection using YOLOv8
- 🗑️ Waste Identification with custom-trained datasets
- 🥊 Violence Detection using RWF-2000 video dataset
- 🔔 Real-Time Alerts via WebSockets
- 📷 CCTV Integration for Live Video Feed

---

## 🔎 Output

- 🎯 Detected objects highlighted with bounding boxes in real-time CCTV feed
- 📊 Alerts generated instantly for:
  - Violence escalation
  - Waste/trash presence
  - High crowd density
- 🖥️ Visual dashboard or CLI output
- 📥 Logs and alerts stored in a centralized system (Firebase/PostgreSQL)

![image](https://github.com/user-attachments/assets/d6c224cb-5802-44b2-839a-a431f63fc1ce)



![image](https://github.com/user-attachments/assets/c116aa27-a706-492a-bc69-717d23308298)



![image](https://github.com/user-attachments/assets/5d08bc3c-e236-4443-8ca5-8ee176df8745)


---

## 🖼️ System Architecture

![image](https://github.com/user-attachments/assets/92c1c63b-af45-451b-a9e4-aaf41011c56c)


