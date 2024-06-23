import time
# Imports the time module for adding delays
from dotenv import load_dotenv
# Imports the load_dotenv function to load environment variables from a .env file
import os
# Imports the os module for interacting with the operating system
import streamlit as st
# Imports the Streamlit library for creating web apps
from langchain_community.vectorstores import FAISS
# Imports FAISS vector store from LangChain for efficient similarity search
from langchain_community.embeddings import HuggingFaceEmbeddings
# Imports HuggingFaceEmbeddings for text embeddings
from langchain.prompts import PromptTemplate
# Imports PromptTemplate for creating customizable prompts
from langchain.memory import ConversationBufferWindowMemory
# Imports ConversationBufferWindowMemory for maintaining conversation history
from langchain.chains import ConversationalRetrievalChain
# Imports ConversationalRetrievalChain for creating a conversational AI system
from langchain.chat_models import ChatOpenAI
# Imports ChatOpenAI for interfacing with OpenAI's chat models
load_dotenv()
# Loads environment variables from a .env file
from footer import footer
# Imports a custom footer function
st.set_page_config(page_title="Law-GPT", layout="centered")
# Sets up the Streamlit page configuration
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("images/banner.png", use_column_width=True)
# Creates a column layout and displays a banner image in the middle column

def hide_hamburger_menu():
    # Function to hide Streamlit's default menu and footer
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()
# Calls the function to hide the menu and footer

@st.cache_resource
def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

# Load the embeddings
embeddings = load_embeddings()

@st.cache_resource
def load_vector_store():
    """Load and cache the vector store."""
    return FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)

# Load the vector store
db = load_vector_store()

# Create the retriever
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# ... (in the main logic)
# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.initial_prompt = True
    st.session_state.section = None
# Initializes session state variables for storing messages and tracking conversation state

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
# Initializes conversation memory in the session state

result = qa.invoke(input=input_prompt)
message_placeholder = st.empty()
answer = extract_answer(result["answer"])

full_response = "⚠️ **_Gentle reminder: Do seek professional help if needed ,i am in no way an alternative for a professional._.** \n\n\n"
print(answer)
if answer:  # Add this check
    for chunk in answer.split():  # Split the answer into words
        full_response += chunk + " "
        time.sleep(0.0001)
        message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)
else:
    full_response += "I'm sorry, I couldn't generate a response. Please try again."
    message_placeholder.markdown(full_response, unsafe_allow_html=True)

# Prompt templates for different sections
mental_health_assistant_prompt = """
<s>[INST]
As a compassionate and knowledgeable mental health assistant, your role is to provide caring and insightful support to individuals seeking guidance or assistance with their emotional well-being. Tailor your responses to meet these objectives:

-Establish a warm and empathetic tone, creating a safe and non-judgmental space for open communication.
-Respond in a conversational and approachable manner, avoiding overly clinical or detached language.
-Actively listen to the individual's concerns, validate their feelings, and offer reassurance when appropriate.
-Encourage self-reflection and self-discovery by asking thoughtful questions and prompting deeper exploration of emotions and experiences.
-Provide practical coping strategies, stress management techniques, and healthy lifestyle recommendations tailored to the individual's unique situation.
-Suggest helpful resources, such as hotlines, support groups, or reputable online resources, when relevant.
-Maintain appropriate boundaries and refrain from providing medical advice or making definitive diagnoses.
-Prioritize the individual's well-being and safety, gently encouraging professional help when necessary.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
-[Express empathy and validate the individual's feelings or experiences]
-[Offer a caring and supportive perspective on the situation]
-[Provide practical coping strategies or self-care recommendations relevant to the concern]
-[Suggest helpful resources or support options, if appropriate]
-[Gently encourage seeking professional help if the situation warrants it]
-[Conclude with a reassuring and compassionate message, reinforcing your availability for further support]
</s>[INST]
"""

mental_health_counselor_prompt = """
<s>[INST]
As a compassionate mental health counselor, your role is to provide a supportive and non-judgmental space for individuals to explore their thoughts, feelings, and experiences. Tailor your approach to meet these objectives:

-Establish a warm, empathetic, and patient tone to create an environment of trust and understanding.
-Listen actively and attentively, allowing the individual to express themselves freely without interruption.
-Validate the individual's emotions and experiences, acknowledging the legitimacy of their perspectives and concerns.
-Ask open-ended questions to encourage self-reflection, deeper exploration, and insight into underlying thoughts and patterns.
-Provide a safe space for the individual to process difficult emotions, offering reassurance and coping strategies when appropriate.
-Gently challenge unhelpful thought patterns or behaviors, offering alternative perspectives and encouraging self-awareness.
-Suggest practical tools and techniques for managing stress, anxiety, or other mental health challenges, tailored to the individual's needs.
-Maintain appropriate boundaries, refraining from giving medical advice or making diagnoses.
-Prioritize the individual's well-being and safety, recommending professional support when necessary.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
-[Reflect the individual's feelings and experiences with empathy and validation]
-[Ask an open-ended question to promote self-exploration and insight]
-[Offer a caring perspective or reframe unhelpful thought patterns]
-[Suggest a relevant coping strategy, relaxation technique, or self-care practice]
-[Gently encourage seeking professional support if the situation warrants it]
-[Conclude with a supportive and reassuring message, reinforcing your non-judgmental presence]
</s>[INST]
"""
def initialize_qa(section):
    # Function to initialize the ConversationalRetrievalChain based on the section
    prompt_template = mental_health_assistant_prompt if section == "advisory" else mental_health_counselor_prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
api_key = os.getenv('OPEN_API_KEY')
# Retrieves the OpenAI API key from environment variables

llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1024)
# Initializes the ChatOpenAI model with specific parameters
def extract_answer(full_response):
    """Extracts the answer from the LLM's full response by removing the instructional text."""
    # For now, let's just return the full response
    return full_response


def reset_conversation():
    # Function to reset the conversation state
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.initial_prompt = True
    st.session_state.section = None

# Display initial prompt for section choice
if st.session_state.initial_prompt:
    with st.chat_message("assistant"):
        st.markdown("**Please choose an option to begin:**\n\n1. Therapy Session \n2. Councelling Session")
    st.session_state.initial_prompt = False

# Handle user's choice
# Handle user's choice
if st.session_state.section is None:
    input_prompt = st.chat_input("Choose an option (1 or 2):")
    if input_prompt:
        if input_prompt == "1":
            st.session_state.section = "advisory"
            st.session_state.messages.append({"role": "assistant", "content": "You have chosen: Therapy session"})
        elif input_prompt == "2":
            st.session_state.section = "ipc"
            st.session_state.messages.append({"role": "assistant", "content": "You have chosen: Dumb questions"})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Invalid choice. Please choose either 1 or 2."})
            st.experimental_rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

else:
    # Initialize qa here, after the section has been chosen
    qa = initialize_qa(st.session_state.section)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    input_prompt = st.chat_input("Ask your question...")
    if input_prompt:
        with st.chat_message("user"):
            st.markdown(f"**You:** {input_prompt}")

        st.session_state.messages.append({"role": "user", "content": input_prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking 💡..."):
                result = qa.invoke(input=input_prompt)
                # ... rest of your code ...
                message_placeholder = st.empty()
                answer = extract_answer(result["answer"])

                full_response = "⚠️ **_Gentle reminder: Do seek professional help if needed ,i am in no way an alternative for a professional._.** \n\n\n"
                print(answer)
                for chunk in answer:
                    full_response += chunk
                    time.sleep(0.0001)
                    message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})

            if st.button('🗑️ Reset All Chat', on_click=reset_conversation):

                st.experimental_rerun()
footer()
