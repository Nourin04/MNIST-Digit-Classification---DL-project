

##  MNIST Digit Classification using Deep Learning

This project is a deep learning-based web application that classifies handwritten digits (0–9) using a trained Convolutional Neural Network (CNN) on the MNIST dataset. The application allows users to upload digit images and receive predictions in real-time via a Streamlit interface.

---

### 📂 Dataset

- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Source:** [Keras Datasets](https://keras.io/api/datasets/mnist/)
- **Size:**
  - Training images: 60,000
  - Test images: 10,000
- **Image Shape:** 28x28 pixels, grayscale
- **Classes:** 10 (Digits 0 through 9)
- **Format:** Each pixel is a value between 0 and 255, which is normalized during preprocessing.

---

### 🛠️ Tech Stack

| Component         | Description                        |
|------------------|------------------------------------|
| **Frontend**      | Streamlit Web App                  |
| **Model**         | CNN built using TensorFlow/Keras   |
| **Language**      | Python                             |
| **Deployment**    | Streamlit locally                  |
| **Development**   | Google Colab + VS Code             |

---

### 📊 Model Architectures Tested

| Model Type                  | Test Accuracy |
|----------------------------|---------------|
| Dense-Only Neural Network  | 97.44%        |
| Basic CNN                  | 98.69%        |
| Deeper CNN + Dropout       | **99.23%**    |

The final deployed model is the **Deeper CNN with Dropout** due to its superior generalization performance.

---

### ✅ Final Model Summary

- **Input Shape:** (28, 28, 1)
- **Layers Used:**
  - Convolutional Layers (Conv2D)
  - MaxPooling
  - Dropout
  - BatchNormalization (optional)
  - Dense (Fully Connected Layer)
- **Activation:** ReLU (intermediate), Softmax (output)
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Regularization:** Dropout

---

### 📈 Performance Metrics

- **Accuracy on Test Set:** 99.23%
- **Loss:** ~0.02 (low loss indicates a well-generalized model)
- **Confidence Score:** Displayed on prediction (Softmax probabilities)

---

### 💡 Features

- Real-time prediction on uploaded digit images
- Automatically preprocesses uploaded images (grayscale, resized, inverted)
- Displays confidence for each class (0–9)
- Clean and minimal UI using Streamlit

---

### ⚠️ Limitations

- The model expects MNIST-like clean digit images. Hand-drawn digits from touchpads or paper scans may result in misclassification.
- Images must be **28x28 grayscale** and centered for highest accuracy.
- Doesn't support direct camera input or drawing canvas (can be added).
- Still confuses similar shapes occasionally (e.g., 4 vs. 9, 2 vs. 7).

---

### 📁 Files and Structure

```
├── app.py                    # Streamlit web app
├── mnist_digit_classifier.keras  # Trained model file
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── MNIST digit classification- DL project.ipynb  # Model training file
```

---

### 📦 Installation & Usage

#### 🔧 Setup
```bash
pip install -r requirements.txt
```

#### 🚀 Run the App
```bash
streamlit run app.py
```

#### 💻 Dependencies
```txt
streamlit
tensorflow>=2.16
numpy>=1.23
Pillow
matplotlib
```

---

### 🌟 Future Improvements

- Add drawing canvas using `streamlit-drawable-canvas`
- Improve robustness using advanced augmentations
- Add camera input support
- Deploy on Hugging Face Spaces / Streamlit Cloud
- Add digit explanation using Grad-CAM (visual explanation of predictions)

---

### 🙌 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- Developed with guidance from OpenAI’s ChatGPT.

