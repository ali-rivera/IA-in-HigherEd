"""
UI.py

Create a (local) user interface using StreamLit (that eventually connects to the pipeline)

Author: Ali Rivera ali.rivera@virginia.edu
8/16/24
"""
#########################################
############ Import packages ############
#########################################
import streamlit as st
import os
from openai import OpenAI


#########################################
############ Configure page #############
#########################################

# Set page title 
st.set_page_config(page_title="DS3001 Virtual Assistant") #in tab
st.title('DS3001 Virtual Assistant') #on page

# Subheader message displayed on page
st.subheader("Hi, I'm the Virtual Assistant for DS3001! You can ask me anything about the course - I'll reference the course content whenever possible. If you're asking about a specific assignment, select that from the left menu so I can best help you! 🤓")

# Create radio list on side bar
with st.sidebar:
    option = st.radio("Questions on a specific assignment? Select from the menu below.", ["General Questions", "Lab 7", "Lab 8", "Lab 9", "Lab 10"])

st.write("You selected:", option) #display selection from radio list

############################################
############## Set up Chatbot ##############
# from streamlit tutorial https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart
############################################

# initiate connection with api stored in .env
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

 # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})