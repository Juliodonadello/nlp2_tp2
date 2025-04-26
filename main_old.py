# main.py

import os
import io
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyPDF2 import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st
from thefuzz import fuzz

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from openai import OpenAI
from langchain_core.runnables import Runnable
from typing import Any, Optional
from langchain_core.runnables.config import RunnableConfig

# --- CONFIGURACIÓN INICIAL ---
load_dotenv()
with open("API_KEYS.txt") as f:
    for line in f:
        key_value = line.strip().split("=")
        if len(key_value) == 2:
            key, value = key_value
            os.environ[key] = value

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

class CustomAzureChatModel(Runnable):
    def __init__(self, model="gpt-4o-mini", temperature=0.5):
        self.model = model
        self.temperature = temperature

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        messages_for_api = []
        
        for message in input.messages:
            if message.type == "system":
                messages_for_api.append({"role": "system", "content": message.content})
            elif message.type == "human":
                corrected_content = message.content.replace("Donadelo", "Donadello") if fuzz.ratio("Donadelo", "Donadello") > 80 else message.content
                messages_for_api.append({"role": "user", "content": corrected_content})
            elif message.type == "ai":
                messages_for_api.append({"role": "assistant", "content": message.content})

        response = client.chat.completions.create(
            messages=messages_for_api,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            top_p=1,
        )

        return response.choices[0].message.content

# --- FUNCIONES AUXILIARES ---

def init_langchain_clients(index_name):
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MPNet-base-v2"
    )
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)

def download_document(file_name):
    with open(file_name, 'rb') as f:
        pdf_content = f.read()

    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    text = text.replace("\n", " ").replace("\\n", " ")
    text = " ".join(text.split())
    return text

def split_text_with_langchain(text, max_length, chunk_overlap, source):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    documents = text_splitter.create_documents([text])
    for doc in documents:
        doc.metadata = {"source": source}
    indices = [f"{source}_{i+1}" for i in range(len(documents))]
    return documents, indices

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

def decisor_node(question):
    name_pattern = r'\b(Julio|Jose|Carlos)\b'
    matches = re.findall(name_pattern, question, re.IGNORECASE)
    selected_names = {match.capitalize() for match in matches}
    return {"plan": list(selected_names)}

# --- PROMPTS ---

DECISOR_PROMPT = """Eres un decisor encargado de elegir un index de pinecone en función del CV consultado.
Las opciones de cv pueden ser Julio, Jose o Carlos. Puede haber selección múltiple.
Siempre se va a especificar explícitamente el nombre del CV consultado."""

JULIO_PROMPT = """Eres un analista de CVs encargado de responder preguntas sobre el CV de Julio.
Si el usuario proporciona una pregunta sobre la cual no tienes contexto, di explícitamente: 'No hay información relevante en el CV'.
Utiliza toda la información a continuación según sea necesario:

------

{context}"""

JOSE_PROMPT = """Eres un analista de CVs encargado de responder preguntas sobre el CV de Jose.
Si el usuario proporciona una pregunta sobre la cual no tienes contexto, di explícitamente: 'No hay información relevante en el CV'.
Utiliza toda la información a continuación según sea necesario:

------

{context}"""

CARLOS_PROMPT = """Eres un analista de CVs encargado de responder preguntas sobre el CV de Carlos.
Si el usuario proporciona una pregunta sobre la cual no tienes contexto, di explícitamente: 'No hay información relevante en el CV'.
Utiliza toda la información a continuación según sea necesario:

------

{context}"""

MIXER_PROMPT = """Eres un experto de CVs encargado de responder preguntas sobre uno o más CVs.
Se te van a proporcionar el contexto de una o más personas dependiendo de la pregunta.
Si el usuario proporciona una pregunta sobre la cual no tienes contexto, di explícitamente: 'No hay información relevante en el CV'.
Utiliza toda la información a continuación según sea necesario:
------

{context}"""

# --- CREACIÓN DE AGENTES ---

def crear_agente_cv(index_name, system_prompt, cv_file=None, source=None, is_mixer=False):

    vectorstore = init_langchain_clients(index_name)
    
    if is_mixer:
        cargar_embeddings_mixer_si_no_existen(vectorstore)
    elif cv_file and source:
        cargar_embeddings_si_no_existen(vectorstore, cv_file, source)
    
    
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25})
    llm = CustomAzureChatModel(model="gpt-4o-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="historial_chat"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

# --- MAIN APP ---

def main_request():
    st.title("Asistente de CV Profesional")
    st.write("Especialista en múltiples CVs (Julio, Jose, Carlos)")

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=3, return_messages=True, memory_key="historial_chat"
        )

    agente_julio = crear_agente_cv(
        index_name='julio-index',
        system_prompt=JULIO_PROMPT,
        cv_file='cv/cv_julio_donadello.pdf',
        source='Julio'
    )

    agente_jose = crear_agente_cv(
        index_name='jose-index',
        system_prompt=JOSE_PROMPT,
        cv_file='cv/cv_jose_martinez.pdf',
        source='Jose'
    )

    agente_carlos = crear_agente_cv(
        index_name='carlos-index',
        system_prompt=CARLOS_PROMPT,
        cv_file='cv/cv_carlos_garcia.pdf',
        source='Carlos'
    )

    agente_mixer = crear_agente_cv(
        index_name='mixer-index',
        system_prompt=MIXER_PROMPT,
        is_mixer=True 
    )

    pregunta = st.text_input("Haz tu pregunta:")

    if pregunta:
        pregunta_procesada = pregunta.replace("Donadelo", "Donadello") if fuzz.ratio("Donadelo", "Donadello") > 80 else pregunta
        
        memory_variables = st.session_state.memory.load_memory_variables({})
        inputs = {
            "input": pregunta_procesada,
            "historial_chat": memory_variables["historial_chat"]
        }

        resultado_decisor = decisor_node(pregunta_procesada)
        nombres_detectados = resultado_decisor['plan']

        if len(nombres_detectados) == 1:
            nombre = nombres_detectados[0]
            if nombre == 'Julio':
                respuesta = agente_julio.invoke(inputs)
            elif nombre == 'Jose':
                respuesta = agente_jose.invoke(inputs)
            elif nombre == 'Carlos':
                respuesta = agente_carlos.invoke(inputs)
            else:
                respuesta = {"answer": "Nombre no reconocido."}
        elif len(nombres_detectados) > 1:
            respuesta = agente_mixer.invoke(inputs)
        else:
            respuesta = {"answer": "No se especificó ningún CV en la pregunta."}

        st.markdown(f"**Respuesta:** {respuesta['answer']}")
        st.session_state.memory.save_context({"input": pregunta}, {"output": respuesta["answer"]})
        st.divider()
        st.caption("Detalles técnicos:")
        st.write(respuesta)

if __name__ == "__main__":
    main_request()
