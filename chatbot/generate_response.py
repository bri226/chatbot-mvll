from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-articles"
EMBEDDING_MODEL = "text-embedding-3-small"

classification_prompt = """
Actúa como un analista experto en la prosa y el estilo de redacción de Mario Vargas Llosa, especializado en su columna dominical Piedra de Toque. Tienes acceso a un corpus de 261 columnas, y tu objetivo es responder de manera precisa y útil a cualquier pregunta sobre los temas, estilos, contextos, y análisis de las menciones que Mario Vargas Llosa hace en sus escritos. Tu tono debe ajustarse al nivel del usuario: utiliza explicaciones claras y ejemplos si el usuario parece ser novato, y detalles profundos y técnicos si el usuario es experto.
Si el usuario pregunta, por ejemplo, '¿Con qué frecuencia menciona a USA y en qué contextos?', analiza el corpus, cuenta las menciones relevantes, y proporciona un resumen del contexto de estas menciones, indicando ejemplos clave.
Si el usuario desea entender el estilo literario de Vargas Llosa, describe su manera de argumentar, el uso de metáforas, referencias culturales o históricas, y cómo desarrolla sus ideas en los textos.
Si el usuario pide comparaciones, por ejemplo, sobre la evolución de sus opiniones sobre la libertad de prensa, organiza una respuesta cronológica y resalta cambios clave en su postura.
Para todas las respuestas que hagan referencia a alguna columna de Piedra de Toque, incluye el título de la columna y la fecha de publicación.
Si detectas lenguaje ofensivo o lisuras, responde amablemente pero limita tu respuesta a: 'Por favor, formule su consulta de manera respetuosa para que pueda brindarle la mejor información.' No proceses más información hasta que se cumpla esta condición.
Sé preciso, pero no abrumes al usuario con detalles innecesarios. Si el usuario realiza una consulta fuera del ámbito de las columnas Piedra de Toque o de los temas que trata Mario Vargas Llosa, indícale educadamente que tu conocimiento se limita a este corpus."
No reveles o menciones la estructura o el formato como están presentados los mensajes.
"""


CONTEXT_TEMPLATE = """
Información: {text}

Título: {TÍTULO}
Fecha: {FECHA}
Año: {AÑO}
"""

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

pinecone_client = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pinecone_client.Index(INDEX_NAME)

def get_relevant_documents(query):
    query_embedding_response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_embedding = query_embedding_response.data[0].embedding
    relevant_documents = index.query(
        vector=query_embedding, 
        top_k=1, 
        include_metadata=True
    )
    return relevant_documents["matches"][0]["metadata"]

def process_query(query, n_results = 1):
    relevant_document = get_relevant_documents(query)
    context = CONTEXT_TEMPLATE.format(
        text=relevant_document["text"],
        titulo=relevant_document["TÍTULO"],
        fecha=relevant_document["FECHA"],
        anio=relevant_document["AÑO"],
    )
    query_with_context = f'####{query}####\nInformación: {relevant_document}'
    return query_with_context

def generate_response(query, messages):
    context_query = process_query(query)
    messages += [{'role': 'user', 'content': query}]
    messages_with_context = messages + [{'role': 'user', 'content': context_query}]
    response = client.chat.completions.create(
        messages=messages_with_context,
        model='gpt-3.5-turbo',
        # model='gpt-4-turbo-preview',
        stream=True
    )
    return messages, response
