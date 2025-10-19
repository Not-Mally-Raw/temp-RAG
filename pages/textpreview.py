import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from extractors import text

st.title("Text Preview")

if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] is not None:
    text_extract = st.button("Extract Text from PDF")
    if text_extract:
        pdf_file: UploadedFile = st.session_state["uploaded_file"]

        # Reset file stream position before reading
        pdf_file.seek(0)
        pdf_contents = pdf_file.read()

        with st.spinner("Extracting sentences from the PDF...", show_time=True):
            text_sentences = text.extract_sentences(pdf_contents)
        st.success("Sentences extracted successfully!")
        st.write("### Extracted Sentences")
        for sentence in text_sentences:
            st.text(sentence)

        # st.warning(f"Type of text: {type(text_sentences)}")
        # st.warning(f"Type of list element: {type(text_sentences[0])} ")
        st.session_state["text"] = text_sentences
        # print("Extracted text:", text_sentences)
else:
    st.info("Please upload a PDF file to continue.")
