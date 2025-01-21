import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatMessagePromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracking\
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "True"
os.environ['LANGCHAIN_PROJECT']="Q&A Chatbot With OPENAI"

# Prompt Template
prompt= ChatMessagePromptTemplate.from_message([
    ("system","You are the helpful assistant. You are here to help the user with any questions they have. You can answer questions about any topic")
    ("user","Question:{Question}")
])

def generate_response(question,api_key,llm,temp,max_tokens):
    openai.api_key = api_key
    llm =ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt |llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

#Title of the app
st.title('Enhaanced Q&A Chatbot With OpenAI')

#Sidebar for the Settigs
st.sidebar.tittle("Settings")
api_key =st.sidebar.text_input("Enter your OPENAI API KEY:",type="password")

## DropDown to choose various llm models
llm=st.sidebar.selectbox("Select a openAI model",["gpt-4o","gpt-4-turbo","gpt-4"])

# Adjust response parameters
temperature= st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_values=50,max_values=300,value=150)

#Main interface for user input
st.write("Go Ahead and ask a question ")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Give Your Query")

    
     







