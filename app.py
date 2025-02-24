import streamlit as st
from agent import Agent
from index_manager import IndexManager
from constants import embed_model, llm_model

@st.cache_resource
def initialize_agent():
    index_manager = IndexManager(embed_model)
    index = index_manager.retrieve_index()
    return Agent(index, llm_model)

#initialize the session state and the agent
if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Arxive Paper ChatBot")

# display the chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about research papers"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.agent.chat(prompt).response
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})