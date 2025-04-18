import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model('mnist_digit_classifier.keras')

# App title
st.title("MNIST Handwritten Digit Classifier")
st.write("Upload an image of a digit (0–9) and the model will try to predict it.")

# File uploader
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("L")           # Convert to grayscale
    image = ImageOps.invert(image)                           # Invert: white digit on black
    image = image.resize((28, 28))                           # Resize to MNIST format
    img_array = np.array(image).astype("float32") / 255.0    # Normalize to [0, 1]
    img_array = np.where(img_array > 0.5, 1.0, 0.0)          # Binarize
    img_reshaped = img_array.reshape(1, 28, 28, 1)            # Add batch & channel dims

    # Display preprocessed image
    st.subheader("🖼️ Preprocessed Image")
    fig, ax = plt.subplots()
    ax.imshow(img_array, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    # Predict using the model
    prediction = model.predict(img_reshaped)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Show results
    st.success(f"✅ Predicted Digit: **{predicted_digit}**")
    st.info(f"📈 Confidence: {confidence:.2f}%")

    # Show full probability distribution (optional)
    st.subheader("🔢 Prediction Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{i}: {prob:.4f}")
