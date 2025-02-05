import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArvixAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv


#Arvix tools
arvix_wrapper = ArvixAPIWrapper(top_k_results=5,doc_content_chars_max=200)
arvix=ArxivQueryRun(api_wrapper=arvix_wrapper)

#Wikipedia tools
wikipedia_wrapper=WikipediaAPIWrapper(top_k_results=5,doc_content_chars_max=200)
wikipedia= WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# Search Duckduckgo
search=DuckDuckGoSearchRun(name="search")


st.title("Langchain Chat with Search")


# Sidebar for the your settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")

if "message" not in st.session_state:
    st.session_state["message"]=[
        {
            "role":"assisant","content":"Hi!! I am the Chatbot who can search the web. How can i help you ???.."
        }
    ]


for msg in st.session_state.message:
    st.chat_message(msg["role"].write(msg["content"]))


if prompt:=st.chat_input(placeholder="What is deep Learning and Reinforcement learning"):
    st.session_state.message.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[arvix,wikipedia,search]

    search_agent=initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,tools=tools,llm=llm,handling_pass_error=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.message,callbacks=[st_cb])
        st.session_state.message.append({"role":"assisant","content":"response"})
        st.write(response)







