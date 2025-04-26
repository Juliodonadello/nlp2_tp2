# main.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from agents.prompts import JULIO_PROMPT, JOSE_PROMPT, CARLOS_PROMPT, MIXER_PROMPT
from agents.agent_factory import crear_agente_cv
from utils.common import decisor_node

# --- CONFIGURACIÓN INICIAL ---

# Cargar variables de entorno
load_dotenv()
with open("API_KEYS.txt") as f:
    for line in f:
        key_value = line.strip().split("=")
        if len(key_value) == 2:
            key, value = key_value
            os.environ[key] = value

# --- STREAMLIT APP ---

def main_request():
    st.title("Asistente de CV Profesional")
    st.write("Especialista en múltiples CVs (Julio, Jose, Carlos)")

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=3, return_messages=True, memory_key="historial_chat"
        )

    # Crear agentes (cargando embeddings si faltan)
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

    # Entrada de usuario
    pregunta = st.text_input("Haz tu pregunta:")

    if pregunta:
        pregunta_procesada = pregunta.replace("Donadelo", "Donadello")  # Corrección automática
        
        memory_variables = st.session_state.memory.load_memory_variables({})
        inputs = {
            "input": pregunta_procesada,
            "historial_chat": memory_variables["historial_chat"]
        }

        # Decidir a qué agente enviar la pregunta
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
