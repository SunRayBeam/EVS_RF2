# EVS_RF2 🎧🌿
**EVS_RF2** is a lightweight, plug-and-play Python module for **Environmental Sound Classification (ESC)** using a **Random Forest classifier** trained on the ESC-50 dataset. It uses **Librosa-extracted audio features** like MFCC, spectral centroid, bandwidth, and zero-crossing rate.

---

## 📦 Features
- 📊 Trained on **entire ESC-50 dataset (2000 samples, 50 categories)**
- 🔍 Uses classical **Random Forest** model for efficiency and interpretability
- 🎧 Supports real-time prediction on new `.wav` files
- 📁 Modular: import and use as a Python package
- ⚙️ Auto audio feature extraction with **Librosa**

---

## 📚 Dataset Requirement

This repo does **not** host the ESC-50 dataset due to size limits.

Please manually download the dataset from the official source:

📥 [karoldvl/ESC-50 (GitHub)](https://github.com/karoldvl/ESC-50)
