import os
import streamlit as st
import pickle
import time
import torch
from sklearn.preprocessing import normalize
import numpy as np
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from transformers import BertTokenizer, BertModel
from langchain.embeddings.base import Embeddings

# Initialize Groq
api = "YOUR API HERE"
llm = ChatGroq(
    temperature=0.3, 
    groq_api_key=api, 
    model_name="llama-3.1-70b-versatile"
)

class BertEmbeddings(Embeddings):
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using BERT"""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                # Ensure text is a string
                if not isinstance(text, str):
                    text = str(text)
                    
                # Tokenize and get embeddings
                inputs = self.tokenizer(text, 
                                     return_tensors='pt', 
                                     truncation=True, 
                                     padding=True, 
                                     max_length=512)
                outputs = self.model(**inputs)
                
                # Get the mean of the last hidden state as embedding
                last_hidden_state = outputs.last_hidden_state.squeeze(0)
                mean_embedding = torch.mean(last_hidden_state, dim=0)
                embeddings.append(mean_embedding.numpy())
                
        # Convert to numpy array and normalize
        embeddings = np.array(embeddings)
        embeddings = normalize(embeddings)
        return embeddings.tolist()  # Convert to list for LangChain compatibility
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text"""
        return self.embed_documents([text])[0]

st.title("WebQueryBot 🤖 ")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    # Filter out empty URLs
    urls = [url for url in urls if url.strip()]
    if not urls:
        st.error("Please enter at least one URL")
    else:
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            
            # Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
            
            # Create embeddings instance
            embeddings = BertEmbeddings()
            
            # Create FAISS index
            vectorstore_bert = FAISS.from_documents(
                documents=docs,
                embedding=embeddings
            )
            
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)
            
            # Save the FAISS index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_bert, f)
                
            main_placeholder.text("Process completed! You can now ask questions.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                st.write(result["answer"])
                
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
