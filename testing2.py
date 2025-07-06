import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Leaf Disease Detection", layout="centered")
st.title("🌿 Leaf Disease Detection")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_leaf_disease_model_resnet.keras")

model = load_model()

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Bio-friendly fertilizer recommendations dictionary
fertilizer_recommendations = {
    "Apple__Black_rot": {
        "fertilizer": "Neem cake + Trichoderma",
        "advice": "Apply neem cake to improve soil health and suppress fungal spores. Combine with Trichoderma biofungicide around the base."
    },
    "Apple__Cedar_apple_rust": {
        "fertilizer": "Seaweed extract + Potassium bicarbonate",
        "advice": "Use seaweed for overall plant health and apply potassium bicarbonate as a foliar biofungicide."
    },
    "Apple__healthy": {
        "fertilizer": "Vermicompost + Panchagavya",
        "advice": "Maintain healthy growth using vermicompost and foliar spray of Panchagavya every 15 days."
    },
    "Corn_(maize)__Cercospora_leaf_spot_Gray_leaf_spot": {
        "fertilizer": "Trichoderma + Farmyard Manure (FYM)",
        "advice": "Mix Trichoderma in FYM and apply during early stages to reduce fungal growth naturally."
    },
    "Corn_(maize)__Common_rust_": {
        "fertilizer": "Compost tea + Copper soap (organic)",
        "advice": "Spray compost tea for microbial diversity and organic copper soap as a rust suppressant."
    },
    "Corn_(maize)__healthy": {
        "fertilizer": "Azospirillum + Organic Potash",
        "advice": "Apply Azospirillum biofertilizer to boost nitrogen uptake and organic potash to support kernel development."
    },
    "Corn_(maize)__Northern_Leaf_Blight": {
        "fertilizer": "Bacillus subtilis + Jeevamrut",
        "advice": "Spray Bacillus subtilis to inhibit leaf blight and enrich soil with Jeevamrut."
    },
    "Grape__Black_rot": {
        "fertilizer": "Sulphur dust + Neem oil",
        "advice": "Apply sulphur dust around roots and neem oil as foliar spray to control fungal growth."
    },
    "Grape__Esca_(Black_Measles)": {
        "fertilizer": "Biochar + Mycorrhiza",
        "advice": "Incorporate biochar to reduce toxin accumulation and apply mycorrhiza to improve root strength."
    },
    "Grape__healthy": {
        "fertilizer": "Organic potassium + Banana peel compost",
        "advice": "Enhance fruiting using potassium-rich organic matter and banana peel compost tea."
    },
    "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "fertilizer": "Trichoderma + Vermiwash",
        "advice": "Apply Trichoderma to reduce leaf spot and use vermiwash as foliar tonic every 10 days."
    },
    "Potato__Early_blight": {
        "fertilizer": "Neem cake + Pseudomonas fluorescens",
        "advice": "Apply neem cake in rows and foliar spray with Pseudomonas to reduce early blight incidence."
    },
    "Potato__healthy": {
        "fertilizer": "Compost + Rhizobium",
        "advice": "Use well-decomposed compost and Rhizobium to improve tuber development."
    },
    "Potato__Late_blight": {
        "fertilizer": "Garlic extract spray + Wood ash",
        "advice": "Foliar spray with garlic extract and dusting with wood ash helps suppress late blight spores."
    },
    "Tomato__Bacterial_spot": {
        "fertilizer": "Cow urine spray + Neem oil",
        "advice": "Spray diluted cow urine and neem oil to act as natural bactericide and growth enhancer."
    },
    "Tomato__healthy": {
        "fertilizer": "Panchagavya + Bone meal",
        "advice": "Spray Panchagavya fortnightly and mix bone meal into soil for sustained flowering and fruiting."
    },
    "Tomato__Septoria_leaf_spot": {
        "fertilizer": "Serenade (Bacillus-based biofungicide) + Compost tea",
        "advice": "Apply Serenade weekly and use compost tea to build foliar immunity."
    }
}

# Normalize label
def normalize_label(label):
    return label.lower().replace("___", "_").replace("__", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").strip()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        st.error("❌ Please upload a valid image file (.jpg, .jpeg, .png).")
    else:
        # Display image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption="Uploaded Leaf Image", use_container_width=False)

        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        confidence = np.max(prediction) * 100

        st.success(f"🩺 Predicted Disease: **{predicted_label}**")
        st.info(f"📊 Confidence: `{confidence:.2f}%`")

        # Normalize label and match recommendation
        norm_label = normalize_label(predicted_label)
        normalized_recs = {normalize_label(k): v for k, v in fertilizer_recommendations.items()}

        if norm_label in normalized_recs:
            fert = normalized_recs[norm_label]
            st.subheader("🌿 Bio-friendly Fertilizer Recommendation")
            st.markdown(f"**Recommended Fertilizer:** `{fert['fertilizer']}`")
            st.markdown(f"**Usage Advice:** {fert['advice']}")
        else:
            st.warning("⚠️ No fertilizer recommendation found for this label.")

        # Show confidence for all classes
        with st.expander("🔍 Show confidence for all classes (Graph + Table)"):
            confidence_df = pd.DataFrame({
                "Class": [class_labels[i] for i in range(len(prediction[0]))],
                "Confidence (%)": [round(p * 100, 2) for p in prediction[0]]
            }).sort_values(by="Confidence (%)", ascending=False)

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(confidence_df["Class"], confidence_df["Confidence (%)"], color='skyblue')
            ax.set_ylabel("Confidence (%)")
            ax.set_xlabel("Class")
            ax.set_title("Model Confidence per Class")
            plt.xticks(rotation=90)
            st.pyplot(fig)

            # Table
            st.dataframe(confidence_df.reset_index(drop=True))
