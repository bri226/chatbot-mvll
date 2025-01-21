from colorama import Fore
import openai
import streamlit as st
import os
from chatbot.html_template import *
from chatbot.question_classifier import classification_prompt, gr_classify_question
from chatbot.unrelated_questions import gr_unrelated_questions
from chatbot.related_questions import gr_related_questions
from supabase import create_client, Client
import uuid
import time
from env_type import production
from colorama import Fore

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv('OPENAI_API_KEY')
DATABASE_NAME = "mvll-articles"
BOT_INTRODUCTION = "Hola, soy el asistente de las columnas de Piedra de Toque de Mario Vargas Llosa en El Comercio. ¿En qué puedo ayudarte hoy?"

if production:
    supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_API_KEY"]
    )

def insert_data(uuid, message, table = DATABASE_NAME):
    data = {"uuid": uuid, "role": message["role"], "content": message["content"]}
    row_insert = supabase.table(table).insert(data)
    return row_insert

def session_id():
    return str(uuid.uuid4())

def write_message(message):
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("user", avatar=BOT_AVATAR):
            st.markdown(message["content"])

def response_from_query():
    
    # Se verifica si el usuario ha ingresado un mensaje
    if st.session_state.prompt == "":
        return

    # Se muestra el historial de mensajes
    for message in st.session_state.history:
        write_message(message)

    # Se muestra el último mensaje del usuario
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(st.session_state.prompt)

    # Se guarda el mensaje en el historial de mensajes
    messages = st.session_state.history

    print("\nMessages: ",messages)
    # Se clasifica la pregunta del usuario
    messages, response = gr_classify_question(st.session_state.prompt, messages)

    # st.session_state.history = messages
    
    value = response.choices[0].message.content
    
    #response.choices[0]: Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='3', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None)
    print(Fore.RED,"\nPregunta: ", st.session_state.prompt,"\nValue: ", value,Fore.BLACK,"\n")
    if value == "SÍ":
        print("Preguntas sobre las columnas Piedra de Toque")
        messages, response_r = gr_related_questions(st.session_state.prompt, messages)
        st.session_state.history = messages
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            assistant_message = st.write_stream(response_r)

    else:
        print("Preguntas de otro tipo")
        messages, response_ur = gr_unrelated_questions(st.session_state.prompt, messages, value)
        st.session_state.history = messages
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            assistant_message = st.write_stream(response_ur)

    st.session_state.history.append(
        {"role": "assistant", "content": assistant_message}
    )
    messages = st.session_state.history

    # data = {"uuid": st.session_state.session_id, "role": messages[-2]["role"], "content": messages[-1]["content"]} #ELIMINAR
    # {'uuid': '218f1bb4-f837-4771-85d5-534e1d2a795b', 'role': 'user', 'content': '¡Hola! ¿En qué puedo ayudarte hoy con las columnas de Piedra de Toque?'}
    # print(data) #ELIMINAR

    if production:
        insert_data(st.session_state.session_id, messages[-2]).execute()
        insert_data(st.session_state.session_id, messages[-1]).execute()

def main():

    # Inicialización de la sesión, historial de mensajes y stream
    if "session_id" not in st.session_state:
        st.session_state.session_id = session_id()
        
    if "history" not in st.session_state:
        st.session_state.history = [{'role': 'system', 'content': classification_prompt}]

    if "stream" not in st.session_state:
        st.session_state.stream = None
    
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.write(BOT_INTRODUCTION)
    
    # Bucle principal de la aplicación
    if prompt := st.chat_input(
        key="prompt", 
        placeholder="Consulta cualquier pregunta relacionada a la columna Piedra de Toque de Mario Vargas Llosa"
    ):
        response_from_query()

if __name__ == "__main__":
    main()
