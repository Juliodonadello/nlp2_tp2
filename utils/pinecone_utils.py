# utils/pinecone_utils.py

import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils.common import download_document, split_text_with_langchain
import streamlit as st

def init_langchain_clients(index_name):
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MPNet-base-v2"
    )
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)

def cargar_embeddings_si_no_existen(pinecone_vectorstore, file_name, source):
    try:
        index_stats = pinecone_vectorstore._index.describe_index_stats()
        if index_stats["total_vector_count"] == 0:
            text = download_document(file_name)
            documents, indices = split_text_with_langchain(
                text, 
                max_length=500, 
                chunk_overlap=50, 
                source=source
            )
            pinecone_vectorstore.add_documents(documents=documents, ids=indices)
            st.success(f"Embeddings de {source} cargados exitosamente!")
    except Exception as e:
        st.error(f"Error cargando embeddings: {e}")

def cargar_embeddings_mixer_si_no_existen(pinecone_vectorstore):
    try:
        index_stats = pinecone_vectorstore._index.describe_index_stats()
        if index_stats["total_vector_count"] == 0:
            # Cargar Julio
            text_julio = download_document('cv/cv_julio_donadello.pdf')
            documents_julio, indices_julio = split_text_with_langchain(text_julio, 500, 50, 'Julio')

            # Cargar Jose
            text_jose = download_document('cv/cv_jose_martinez.pdf')
            documents_jose, indices_jose = split_text_with_langchain(text_jose, 500, 50, 'Jose')

            # Cargar Carlos
            text_carlos = download_document('cv/cv_carlos_garcia.pdf')
            documents_carlos, indices_carlos = split_text_with_langchain(text_carlos, 500, 50, 'Carlos')

            # Juntar todo
            all_documents = documents_julio + documents_jose + documents_carlos
            all_indices = indices_julio + indices_jose + indices_carlos

            pinecone_vectorstore.add_documents(documents=all_documents, ids=all_indices)
            st.success("Embeddings del Mixer cargados exitosamente!")
    except Exception as e:
        st.error(f"Error cargando embeddings del Mixer: {e}")
