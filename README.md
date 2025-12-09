# smart-ai-mental-health-detection
Deep Learning based Mental Health Detection using Facial Emotion Recognition + Text Sentiment Analysis (FER-2013 Dataset)


This project detects a person's mental/emotional state using:

- Facial Emotion Recognition (FER-2013 Dataset)**
- CNN-based Deep Learning model**
- Text Sentiment Analysis using NLP**
- Risk Level Classification (Low / Medium / High)**
- Doctor-like Feedback + Final Report**

---

## **Project Workflow**

1. Load FER-2013 dataset  
2. Preprocess images (48x48 grayscale)  
3. Train a CNN model  
4. Predict emotion from user image  
5. Perform text sentiment analysis  
6. Combine both → Generate mental health report

---

##  **Google Colab Notebook**
Since model files are larger than 25MB, they are stored in Google Drive.

 **Colab Notebook PDF:**  
https://drive.google.com/file/d/1_WwekA4ASankDRithqeeUgahKxr5r1TF/view?usp=sharing  

(*This contains full code, training graphs, confusion matrix, and results.*)

---

##  **Technologies Used**
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- TextBlob  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

##  **Features**
✔ Detects real-time facial emotions  
✔ Sentiment analysis of user conversation  
✔ Clinical-style mental health report  
✔ Advice + doctor’s responses  
✔ Training accuracy/loss visualization  
✔ Confusion matrix for evaluation  

---

##  **Model Summary (CNN)**
- Conv2D layers  
- BatchNormalization  
- MaxPooling  
- Dropout  
- Dense + Softmax final layer  

Trained on **35,000+ images** from FER-2013.

---

##  **Results**
- Training Accuracy: ~58%  
- Validation Accuracy: ~55%  
- Confusion Matrix included in notebook  

---

##  **How to Run (Simple Version)**
Since model file is large, download it from Drive:

👉 Model Download:  
https://drive.google.com/file/d/1_WwekA4ASankDRithqeeUgahKxr5r1TF/view?usp=sharing  





Then run:

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']
model = load_model("emotion_model.h5")

def predict(img):
    img = cv2.imread(img, 0)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1)/255.0
    pred = model.predict(img)[0]
    print("Emotion:", emotion_labels[np.argmax(pred)])

 
