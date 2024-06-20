# Ashray: AI Mental Health Advice Assistant 🧠💬

Ashray is an advanced mental health advisory chatbot designed to provide empathetic, contextually appropriate responses to mental health queries. It leverages cutting-edge machine learning technologies to process and retrieve mental health information effectively.

---

## Features 🌟

- **Knowledge Base Integration**: Automated processing of mental health resources to build a comprehensive knowledge base.
- **Real-Time Support**: Instant mental health guidance through a user-friendly conversational interface built with Streamlit.
- **Empathetic Response Templating**: Structured prompt format ensuring compassionate, detailed, and accurate responses to mental health concerns.

<br>

---

<h4><strong>🚀 Discover peace of mind! Explore Ashray <a href="https://huggingface.co/spaces/thegovindkrishna/Ashray">here</a>. Start your journey to better mental health today! 🌈</strong></h4>
<br>

---

## Components 🛠️

### Knowledge Base Builder (`build_knowledge.py`)

| Functionality | Description |
|---------------|-------------|
| **Resource Loading** | Imports mental health information from various sources. |
| **Text Processing** | Breaks down resources into manageable, topic-specific segments. |
| **Embedding Creation** | Uses advanced NLP models to generate text embeddings. |
| **Vector Database** | Indexes embeddings for quick and efficient information retrieval. |

### Web Application (`app.py`)

| Feature | Description |
|---------|-------------|
| **Streamlit Interface** | Offers a welcoming web interface for user interaction. |
| **Conversation Management** | Handles chat flow and maintains conversation history. |
| **Information Retrieval** | Utilizes the knowledge base to fetch relevant mental health information. |

---

## Setup 📦

### Prerequisites

- Python 3.8 or later
- langchain
- streamlit
- faiss-cpu
- transformers

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thegovindkrishna/Ashray.git
   cd Ashray
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the AI model API Key:**
   Obtain an API key from your chosen AI model provider (e.g., OpenAI, Anthropic).

   Set the API key as an environment variable:
   
   - On macOS and Linux:
     ```bash
     echo "export AI_MODEL_API_KEY='Your-API-Key-Here'" >> ~/.bash_profile
     source ~/.bash_profile
     ```
   - On Windows (Command Prompt):
     ```cmd
     setx AI_MODEL_API_KEY "Your-API-Key-Here"
     ```
   - On Windows (PowerShell):
     ```powershell
     [Environment]::SetEnvironmentVariable("AI_MODEL_API_KEY", "Your-API-Key-Here", "User")
     ```

## Running the Application

1. **Build the knowledge base:**
   ```bash
   python build_knowledge.py
   ```

2. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Usage 🤝

Navigate to the local URL provided by Streamlit to interact with Ashray. Share your mental health concerns or questions, and receive compassionate, informed responses based on the integrated mental health resources. Use the chat interface to engage in supportive conversations and gain valuable mental health insights.

---

## Disclaimer ⚠️

Ashray is an AI assistant designed to provide general mental health information and support. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the guidance of qualified health providers for any mental health concerns.

---

## Contributing 🤝

We welcome contributions to improve Ashray! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to submit issues, feature requests, and code changes.

---

## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

<p align="center">Made with ❤️ for better mental health</p>

