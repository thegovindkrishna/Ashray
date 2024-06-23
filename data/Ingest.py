
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2  


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading documents...")
loader = DirectoryLoader('data', glob="./*.txt")
documents = loader.load()

logging.info("Extracting and splitting texts from documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = []
for document in documents:
    if hasattr(document, 'get_text'):
        text_content = document.get_text() 
    else:
        text_content = "" 

    texts.extend(text_splitter.split_text(text_content))

def embedding_function(text):
    embeddings_model = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    return embeddings_model.embed_query(text)

index = IndexFlatL2(768)  

docstore = {i: text for i, text in enumerate(texts)}
index_to_docstore_id = {i: i for i in range(len(texts))}

faiss_db = FAISS(embedding_function, index, docstore, index_to_docstore_id)

logging.info("Storing embeddings in FAISS...")
for i, text in enumerate(texts):
    embedding = embedding_function(text)
    faiss_db.add_documents([embedding])

logging.info("Exporting the vector embeddings database...")
faiss_db.save_local("ipc_embed_db")

logging.info("Process completed successfully.")
