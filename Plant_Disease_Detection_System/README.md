An AI-powered web application that detects plant diseases from images using a deep learning model. The system helps farmers and agricultural experts diagnose plant diseases early, ensuring timely treatment and improved crop yield.

🚀 Features<br>
🌱 Upload plant images to detect diseases.<br>
🤖 Deep learning model trained on plant disease datasets.<br>
📊 Provides disease details and recommended treatments.<br>
💻 User-friendly web interface built with Streamlit.<br>
☁️ Deployed on AWS EC2 for global accessibility.<br>
🔒 Secure and scalable deployment.<br>
⚙️ Tech Stack<br>
Frontend: Streamlit<br>
Backend: Python (TensorFlow/Keras)<br>
ML Model: Pre-trained .keras model (stored using Git LFS)<br>
Deployment: AWS EC2<br>
<br>
Structure:-<br>
plant-disease-detection/<br>
│
|
│──── plant_disease_model.keras   # Trained ML model<br>
│<br>
├── Plant_disease_app.py                          # Streamlit frontend application<br>
├── requirements.txt                 # Python dependencies<br>
├── README.md                        # Project documentation<br>
