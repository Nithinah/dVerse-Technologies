# dVerse-Technologies

### 1. Parkinson's Disease Detection (CNN + BiLSTM)
- **Description:**  
   This project uses a combination of Convolutional Neural Networks (CNN) and Bidirectional LSTM (BiLSTM) to detect Parkinson's disease from medical data.  

- **Technologies Used:**  
   - TensorFlow  
   - Keras  
   - Scikit-learn  
   - Keras Tuner  
   - PCA (Principal Component Analysis)  

- **Approach:**  
   - CNN extracts spatial patterns from the input data.  
   - BiLSTM processes the temporal dependencies in the extracted features.  
   - Keras Tuner optimizes hyperparameters such as learning rate, dropout rate, and filter size.  
   - Data augmentation is applied using noise injection for better generalization.  
   - PCA is used for decision boundary visualization.  

- **Results:**  
   - Achieved high accuracy with precise pattern recognition.  
   - High AUC score and clear separation in the decision boundary.  

---

### 2. Hand Landmarks Tracking (MediaPipe + OpenCV)
- **Description:**  
   Developed a computer vision project to track and identify hand landmarks using MediaPipe and OpenCV.  

- **Technologies Used:**  
   - OpenCV  
   - MediaPipe  
   - NumPy  

- **Approach:**  
   - MediaPipe’s Hand Landmarker model detects keypoints (fingertips, joints) in real-time.  
   - OpenCV is used for visualization and labeling of the detected landmarks.  
   - Custom logic is implemented to detect whether the hand is open or closed and identify finger extension.  
   - Smoothing of landmark positions using a moving average filter (buffer size = 5).  

- **Results:**  
   - Achieved real-time tracking with low latency and high accuracy.  
   - Accurate detection of finger states and hand gestures.  

---

### 3. HDFC Hybrid FAQ Chatbot (DistilBERT + GPT-2 Fallback)
- **Description:**  
   Built a hybrid chatbot for HDFC Bank FAQs using DistilBERT for retrieval-based matching and GPT-2 Medium for fallback answers.  

- **Technologies Used:**  
   - DistilBERT (for similarity matching)  
   - GPT-2 Medium (for generative fallback)  
   - FAISS (for fast similarity search)  
   - Streamlit (for UI)  

- **Approach:**  
   - DistilBERT retrieves answers from the FAQ dataset using FAISS-based similarity search.  
   - If the similarity score is low, GPT-2 generates a context-aware response.  
   - Implemented similarity thresholding for switching between retrieval and generative responses.  

- **Results:**  
   - High accuracy in retrieval-based answers with smooth fallback for unknown queries.  
   - Fast response time with accurate similarity matching.  
