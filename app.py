import os
from dotenv import load_dotenv
import requests
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import tempfile

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def welcome_message():
    query_params = st.query_params
    username = query_params.get("username", "")
    return f"Hi, {username}! Welcome to our Customer Support ChatBot for payments and banking! 🏦💳\n\nHow can we assist you today? Whether you have questions about transactions, account balances, or any other banking inquiries, feel free to ask. We're here to help you with any payment-related or banking-related concerns you may have.\n\nSimply type your query in the chatbox below, and we'll provide you with the assistance you need.\n\nLet's get started! 🚀" if username else "Welcome!"

# Hardcoded file path for the CSV file
csv_file_path = "banking payments FAQ.csv"

loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")
data = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectors = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,
                                                                       convert_system_message_to_human=True),
                                              retriever=vectors.as_retriever())
st.header("Customer Support ChatBot")

def conversational_chat(query):
    prompt_template = "Given the user query '{user_query}', can you provide a helpful response related to banking and payments, considering the FAQ data we have?"
    filled_prompt = prompt_template.format(user_query=query)
    
    result = chain({"question": filled_prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def fallback_response():
    # Define your fallback response here
    return "I'm sorry, I didn't understand that. Could you please rephrase your question?"

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = [welcome_message()]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]

# container for the chat history
response_container = st.container()
# container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your CSV data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    # Add a button to close the chatbot session and redirect
  
    if submit_button and user_input:
        if "thank you" in user_input.lower():
            response = "You're welcome!"
        elif "how many records" in user_input.lower():
            response = f"There are {len(data)} records in the file."
        else:
            response = conversational_chat(user_input)
            if response == "0":
                response = fallback_response()

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(response)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
