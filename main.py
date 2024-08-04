from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import ChatOpenAI

#First Load all the information stored in dotenv file.

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Function to load and read the given pdfs and store the raw text.

def read_pdf_text(pdf_docs):
    raw_text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text = raw_text + page.extract_text()
    return raw_text


def read_from_url(url):
    urls =[]
    urls.append(url)
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data

def divide_convert_document_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    docs = text_splitter.split_documents(document)
    # create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # Create embeddings from texts
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore


def divide_convert_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    text_chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    return vector_store
    

def get_conversational_chain():
    prompt = """
    if the answer is not in the provided contest Please inform us 'I cant find the answer from the information 
    you have provided' thats it.Please dont provide wrong answers.
    
    Context/Information :\n {context}\n
    Query/Question : \n {Query}\n

    Answer :
    """

    llm_model = ChatGoogleGenerativeAI(model = "gemini-pro",temperature = 0.3)
    prompt_template = PromptTemplate(
        input_variables = ['context','Query'],
        template = prompt

    )

    chain = load_qa_chain(llm_model,chain_type="stuff",prompt = prompt_template)
    return chain

def user_input_answer(question):    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.load_local("Macintosh HD\\Users\\arnav\\desktop\\Chatbot\\faiss_index", embeddings, allow_dangerous_deserialization = True)
    docs = vector_db.similarity_search(question)

    chain = get_conversational_chain()
    response = chain.invoke({
        "input_documents":docs,
         "Query": question
         }
        , return_only_outputs=True)

    return response["output_text"]







