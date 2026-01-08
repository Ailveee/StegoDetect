import streamlit as st
import joblib
from detector import detect

st.set_page_config(page_title="Steganography Detector", layout="centered")
st.title("🕵️ Steganography Detection App")
st.write("Hybrid detection using statistical analysis + machine learning")

model = joblib.load("stego_model.pkl")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    with open("temp.png", "wb") as f:
        f.write(uploaded.read())

    verdict, score = detect("temp.png", model)

    st.image("temp.png", caption="Uploaded Image", width=600)
    st.subheader(f"Verdict: **{verdict}**")
    st.progress(float(score))
    st.caption(f"Steganography probability: {score}")
