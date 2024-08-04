This project implements a chatbot through the Gemini Pro LLM released by Google.
The Gemini Pro LLM can be implemented into your code by generating an Application Programming Interface (API) key.
This chatbot can be used to answer any user's queries pertaining to the uploaded PDF file or URL. 
The chatbot first reads the text in the user input and splits it into chunks.
These chunks are then converted to numerical vectors and stored in the Facebook AI Similarity Search (FAISS) Database.
Then, the user query is embedded and matched with the numerical vectors in the Database.
The query is matched with the most similar chunks from the database, and the output text is produced.

MODULES/LIBRARIES USED:

requests

streamlit

google-generativeai

python-dotenv

langchain

PyPDF2

faiss-cpu

langchain_google_genai

langchain_community
