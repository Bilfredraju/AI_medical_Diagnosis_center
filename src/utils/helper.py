import streamlit as st

def show_header():

    st.markdown(
        """
# 🧠 AI Medical Diagnosis System

Upload a Brain MRI image for diagnosis.
"""
    )


def show_footer():

    st.markdown("---")

    st.caption(
        "Developed using TensorFlow • FastAPI • Streamlit"
    )