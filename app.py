import streamlit as st
from main import read_pdf_text
from main import divide_convert_document_chunks
from main import divide_convert_text_chunks
from main import user_input_answer
from main import read_from_url


st.set_page_config("Query With Multiple PDF files")
st.header("Chat with Google Gemini Pro LLM Model to query multiple pdfs")

user_query = st.text_input("Ask your query related to your uploaded files...")

if user_query:
    response = user_input_answer(user_query)
    st.write(response)

with st.sidebar:
    st.title("Uploaded Files")
    pdf_docs = st.file_uploader("Upload your files in .pdf format and Click on Submit and process your files",accept_multiple_files=True)
    st.write("OR")
    url = st.text_input("Paste the url of any article..")

    button = st.button("Submit and Process")
    if(button):
        with st.spinner("Processing...."):
            
            if(url):
                document = read_from_url(url)
                vector2 = divide_convert_document_chunks(document)
                vector2.save_local("Macintosh HD\\Users\\arnav\\desktop\\Chatbot\\faiss_index")
            else:
                raw_text = read_pdf_text(pdf_docs)
                vector1 = divide_convert_text_chunks(raw_text)
                vector1.save_local("Macintosh HD\\Users\\arnav\\desktop\\Chatbot\\faiss_index")
            
            st.success("Files Processed. You can continue to Ask your Queries")

