MemeMood 🎭
An Emotion-Based Meme Identifier Using Computer Vision

MemeMood watches your face through a webcam, classifies your visible facial expression using a CNN, and hands you a matching meme in real time — an end-to-end demo of a computer-vision + machine-learning application, from frame capture to a fun, user-facing result.

Webcam → Frame Capture → Face Detection → Preprocessing → Emotion Model → Meme Mapping → Output
✨ Features
Real-time webcam capture and face detection (OpenCV Haar Cascade by default)
CNN-based classification into 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
Configurable emotion → meme category mapping
Modular pipeline — swap the detector, model, or meme source independently
Runs fully locally; no camera frames are stored or transmitted
📁 Project Structure
MemeMood/
├── dataset/                 # training data (not committed — see Setup)
├── memes/                   # local meme library, one folder per emotion
│   ├── happy/  sad/  angry/  surprise/  fear/  disgust/  neutral/
├── models/
│   └── emotion_model.h5     # trained model (not committed — see Setup)
├── src/
│   ├── face_detection.py    # face detection module
│   ├── emotion_detection.py # emotion classification module
│   ├── meme_identifier.py   # emotion → meme mapping + retrieval
│   ├── main.py               # live webcam application (entry point)
│   └── train_model.py        # optional script to train your own model
├── requirements.txt
├── LICENSE
└── README.md
🚀 Setup
Clone and install dependencies

git clone https://github.com/<your-username>/MemeMood.git
cd MemeMood
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
Add a trained model

Train your own on FER-2013 (or a similar labelled dataset), respecting its license:

# after placing images under dataset/train/<class>/ and dataset/test/<class>/
python src/train_model.py
This saves the best checkpoint to models/emotion_model.h5. Without a model present, the app still runs in a stub mode (random predictions) so you can verify the pipeline end-to-end before training.

Add meme images

Drop .jpg / .png / .gif files into the matching folder under memes/ (e.g. memes/happy/). Use only images you have the right to use/redistribute.

Run the app

python src/main.py
Press q in the video window to quit.

🧠 Model
A lightweight CNN trained on 48×48 grayscale face crops:

Layer	Output Shape
Input	48×48×1
Conv2D + ReLU ×2	46×46×32
MaxPooling2D	23×23×32
Conv2D + ReLU ×2	21×21×64
MaxPooling2D + Dropout	10×10×64
Dense + ReLU + Dropout	256
Dense + Softmax	7
See src/train_model.py for the full training configuration.

⚠️ Limitations & Ethics
This project classifies visible facial-expression patterns, not a person's actual internal emotional state — please don't treat it as one.
Accuracy depends on lighting, camera quality, occlusion, and how closely a user's expressions resemble the training data.
All processing is local by default. If you extend this project to store images or build user profiles, add explicit consent and clear data-handling practices first.
Full technical write-up, architecture rationale, evaluation methodology, and future-work roadmap are available in the accompanying project report.

🛣️ Roadmap
 Temporal smoothing across frames for more stable predictions
 Multi-face support
 Searchable/tagged meme database
 Streamlit or web-based interface
 Transfer-learning-based model upgrade
📄 License
Released under the MIT License.
