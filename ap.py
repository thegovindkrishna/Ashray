import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="ASHRAY - Mental Health Assistant", page_icon=":brain:")

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Set up Streamlit page configuration

# Custom CSS to make it look more like ChatGPT
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding: 0 1.5rem;
      color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to create conversation chain
def get_conversation_chain():
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory()
    
    template = """As a compassionate and knowledgeable mental health assistant, your role is to provide caring and insightful support to individuals seeking guidance or assistance with their emotional well-being. Tailor your responses to meet these objectives:
-Establish a warm, empathetic, and patient tone to create an environment of trust and understanding.
-Listen actively and attentively, allowing the individual to express themselves freely without interruption.
-Validate the individual's emotions and experiences, acknowledging the legitimacy of their perspectives and concerns.
-Ask open-ended questions to encourage self-reflection, deeper exploration, and insight into underlying thoughts and patterns.
-Provide a safe space for the individual to process difficult emotions, offering reassurance and coping strategies when appropriate.
-Gently challenge unhelpful thought patterns or behaviors, offering alternative perspectives and encouraging self-awareness.
-Suggest practical tools and techniques for managing stress, anxiety, or other mental health challenges, tailored to the individual's needs.
-Maintain appropriate boundaries, refraining from giving medical advice or making diagnoses.
-Prioritize the individual's well-being and safety, recommending professional support when necessary.
-Respond in a conversational and approachable manner, avoiding overly clinical or detached language.
-Actively listen to the individual's concerns, validate their feelings, and offer reassurance when appropriate.
-Encourage self-reflection and self-discovery by asking thoughtful questions and prompting deeper exploration of emotions and experiences.
-Provide practical coping strategies, stress management techniques, and healthy lifestyle recommendations tailored to the individual's unique situation.
-Suggest helpful resources, such as hotlines, support groups, or reputable online resources, when relevant.
-Maintain appropriate boundaries and refrain from providing medical advice or making definitive diagnoses.
-Prioritize the individual's well-being and safety, gently encouraging professional help when necessary.

ANSWER:
-[Reflect the individual's feelings and experiences with empathy and validation]
-[Ask an open-ended question to promote self-exploration and insight]
-[Offer a caring perspective or reframe unhelpful thought patterns]
-[Suggest a relevant coping strategy, relaxation technique, or self-care practice]
-[Gently encourage seeking professional support if the situation warrants it]
-[Conclude with a supportive and reassuring message, reinforcing your non-judgmental presence]
    
    Current conversation:
    {history}
    Human: {input}
    ASHRAY:"""
    
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    return conversation_chain

# Function to handle user input and generate response
def handle_user_input(user_input, conversation):
    response = conversation.predict(input=user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    return response

# Main function to run the Streamlit app
def main():
    st.header("ASHRAY - Your Mental Health Assistant :brain:")
    st.write("Hello! I'm ASHRAY, your personal mental health assistant. How can I help you today?")

    # Initialize conversation chain
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_input = st.chat_input("Share your thoughts or concerns...")

    if user_input:
        # Generate and display response
        response = handle_user_input(user_input, st.session_state.conversation)
        with st.chat_message("assistant"):
            st.write(response)

    # Add a disclaimer
    st.sidebar.markdown("""
    **Disclaimer:** ASHRAY is an AI assistant and not a substitute for professional mental health care. 
    If you're experiencing a mental health emergency, please contact your local emergency services or 
    a mental health crisis hotline immediately.
    """)

if __name__ == '__main__':
    main()