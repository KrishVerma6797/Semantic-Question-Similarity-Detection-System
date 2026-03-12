import streamlit as st
from src.transformer_model import TransformerSimilarity

st.title("Semantic Question Similarity Detector")

model = TransformerSimilarity()

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Compare"):
    if q1 and q2:
        sim = model.similarity(q1, q2)

        st.write("Similarity score:", round(float(sim), 3))

        if sim > 0.7:
            st.success("Duplicate question")
        else:
            st.error("Different question")
    else:
        st.warning("Please enter both questions")
