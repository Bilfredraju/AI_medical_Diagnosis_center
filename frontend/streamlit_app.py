import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Medical Diagnosis",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Medical Diagnosis System")

st.write(
    "Upload a Brain MRI image for tumor prediction."
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("Predict"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                uploaded_file.type
            )
        }

        response = requests.post(
            API_URL,
            files=files
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction Complete")

            st.subheader("Prediction")

            st.write(result["prediction"])

            st.subheader("Confidence")

            st.write(f"{result['confidence']*100:.2f}%")

        else:

            st.error("Prediction failed.")