import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.rag import HandbookRAG  # noqa: E402

st.set_page_config(page_title="Employee Handbook Assistant")

st.title("Employee Handbook Assistant")

question = st.text_input("Question", placeholder="How often must passwords be changed?")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    try:
        with st.spinner("Searching the handbook..."):
            result = HandbookRAG().ask(question.strip())
        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Source")
        st.write(", ".join(result["sources"]) or "No source found")
        with st.expander("Retrieved context"):
            for index, context in enumerate(result["contexts"], start=1):
                st.markdown(f"**Chunk {index}**")
                st.write(context)
    except Exception as exc:
        st.error(str(exc))
elif ask:
    st.warning("Enter a question first.")
