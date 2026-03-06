import streamlit as st
import numpy as np
import tempfile
import os
from utils.data import load_data
from model.train import train_model
from model.predict import predict_image
 

st.title("Teachable Machine Image Classifier")


class_input = st.text_input("Enter class names (comma separated)")

if class_input:
    class_names = [c.strip() for c in class_input.split(",")]

  
    uploaded_files_by_class = {}

    for class_name in class_names:
        files = st.file_uploader(
            f"Upload images for class '{class_name}'",
            type=["jpg", "png"],
            accept_multiple_files=True,
            key=f"upload_{class_name}"
        )
        if files:
            uploaded_files_by_class[class_name] = files

   
    if st.button("Train Model"):
        if not uploaded_files_by_class:
            st.warning("Please upload files for at least one class")
        else:
            temp_dir = tempfile.mkdtemp()
            images = []
            labels = []

            for class_idx, (class_name, files) in enumerate(uploaded_files_by_class.items()):
                class_folder = os.path.join(temp_dir, class_name)
                os.makedirs(class_folder, exist_ok=True)
                for file in files:
                    file_path = os.path.join(class_folder, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    images.append(file_path)
                    labels.append(class_idx)

            # Load data
            train_data, val_data = load_data(temp_dir)
            num_classes = len(class_names)

            # Train model
            model= train_model(train_data , val_data, num_classes)

            st.success("Training Finished")
            st.session_state["model"] = model
            st.session_state["classes"] = class_names

      
# Step 4: Predict new images

st.header("Predict New Images")

if "model" not in st.session_state or "classes" not in st.session_state:
    st.info("Train a model first to make predictions.")
else:
    uploaded_test_files = st.file_uploader(
        "Upload images to predict",
        type=["jpg", "png"],
        accept_multiple_files=True,
        key="predict_files"
    )

    if uploaded_test_files:
        model = st.session_state["model"]
        class_names = st.session_state["classes"]

        for file in uploaded_test_files:
            st.success("Prediction")
            pred_class, pred_probs = predict_image(model, file, class_names)
            st.image(file, width = 400)
            st.write(f"Predicted : {pred_class}")
            confidence = pred_probs[pred_class] * 100  # convert to percentage
            st.write(f"Confidence: {confidence:.2f}%")