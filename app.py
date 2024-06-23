import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from streamlit import cache_data

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
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "greeting_sent" not in st.session_state:
    st.session_state.greeting_sent = False

@cache_data
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    
    # Combine Context and Response into a single text
    text_data = df.apply(lambda row: f"Context: {row['Context']}\nResponse: {row['Response']}\n\n", axis=1).sum()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text_data)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore

# Function to create conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    template = """As a compassionate and knowledgeable mental health assistant cum therapist, your role is to provide caring and insightful support to individuals seeking guidance or assistance with their emotional well-being. Tailor your responses to meet these objectives:
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


ANSWER:
-[Reflect the individual's feelings and experiences with empathy and validation]
-[Ask an open-ended question to promote self-exploration and insight]
-[Offer a caring perspective or reframe unhelpful thought patterns]
-[Suggest a relevant coping strategy, relaxation technique, or self-care practice]
-[Gently encourage seeking professional support if the situation warrants it]
-[Conclude with a supportive and reassuring message, reinforcing your non-judgmental presence]

    Relevant information:
    {context}

    Current conversation:
    {chat_history}
    Human: {question}
    ASHRAY:"""
    
    prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

# Function to handle user input and generate response
def handle_user_input(user_input):
    if st.session_state.conversation is None:
        st.error("Conversation not initialized. Please try refreshing the page.")
        return
    response = st.session_state.conversation({"question": user_input})
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    return response["answer"]
# Function to send greeting
def send_greeting():
    greeting = "Hello! I'm ASHRAY, your personal mental health assistant. How can I help you today?"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.greeting_sent = True


# Main function to run the Streamlit app
def main():
    st.header("ASHRAY - Your Mental Health Assistant :brain:")

    csv_path = "train.csv"  # Make sure this file is in the same directory as your script
    vectorstore = load_csv_data(csv_path)

    # Sidebar
    st.sidebar.title("Chat Options")
    if st.sidebar.button("New Chat"):
        st.session_state.messages = []
        st.session_state.conversation = None
        st.session_state.greeting_sent = False
        st.experimental_rerun()

    # Send greeting if not already sent
    if not st.session_state.greeting_sent:
        send_greeting() 
            
    # Initialize conversation chain
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(vectorstore)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_input = st.chat_input("Share your thoughts or concerns...")

    if user_input:
        # Generate and display response
        response = handle_user_input(user_input)
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