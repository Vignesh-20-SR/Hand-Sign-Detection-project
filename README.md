# Hand Sign Detection ✋🤖

A Machine Learning project that detects and classifies hand signs (A–Z) using **American Sign Language (ASL)** with **MediaPipe**, **OpenCV**, and **scikit-learn**. 

## 🚀 Features
- Real-time hand tracking with MediaPipe
- Image dataset collection using webcam
- Custom dataset preprocessing
- Machine learning model training
- Hand sign prediction in live video

## 📂 Project Structure
📦HandSignDetection
┣ 📜collect_img.py # Capture hand sign images
┣ 📜create_dataset.py # Preprocess dataset
┣ 📜train_classifier.py # Train ML classifier
┣ 📜interference_classifier.py # Run predictions
┣ 📜verify_ds.py # Dataset verification
┣ 📜requirements.txt # Dependencies
┗ 📜README.md # Documentation


## ⚙️ Installation
```bash
git clone https://github.com/your-username/HandSignDetection.git
cd HandSignDetection
pip install -r requirements.txt
```
## ▶️ Usage

- Collect Images
```bash
python collect_img.py
```
- Create Dataset
```bash
python create_dataset.py
```
- Train Classifier
```bash
python train_classifier.py
```

- Run Live Prediction
```bash
python interference_classifier.py
```
## 📊 Tech Stack
- Python 3.11
- OpenCV
- MediaPipe
- scikit-learn
- NumPy
- Matplotlib

## 📌 Note
- Dataset (data/, data_3/) is not included in this repo.
- You can regenerate your own dataset using collect_img.py.

Made with ❤️ by Vignesh
