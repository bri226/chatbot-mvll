from openai import OpenAI
import streamlit as st
from chromadb import PersistentClient
from vector_store import openai_embedding

classification_prompt = """
Actúa como un analista experto en la prosa y el estilo de redacción de Mario Vargas Llosa, especializado en su columna dominical Piedra de Toque. Tienes acceso a un corpus de 261 columnas, y tu objetivo es responder de manera precisa y útil a cualquier pregunta sobre los temas, estilos, contextos, y análisis de las menciones que Mario Vargas Llosa hace en sus escritos. Tu tono debe ajustarse al nivel del usuario: utiliza explicaciones claras y ejemplos si el usuario parece ser novato, y detalles profundos y técnicos si el usuario es experto.
Además, sigue estas reglas al interactuar:
1. Preguntas sobre menciones o temas específicos: Si el usuario pregunta, por ejemplo, '¿Con qué frecuencia menciona a USA y en qué contextos?', analiza el corpus, cuenta las menciones relevantes, y proporciona un resumen del contexto de estas menciones, indicando ejemplos clave.
2. Estilo de redacción y enfoque: Si el usuario desea entender el estilo literario de Vargas Llosa, describe su manera de argumentar, el uso de metáforas, referencias culturales o históricas, y cómo desarrolla sus ideas en los textos.
3. Preguntas comparativas o críticas: Si el usuario pide comparaciones, por ejemplo, sobre la evolución de sus opiniones sobre la libertad de prensa, organiza una respuesta cronológica y resalta cambios clave en su postura.
4. Manejo de lenguaje inapropiado: Si detectas lenguaje ofensivo o lisuras, responde amablemente pero limita tu respuesta a: 'Por favor, formule su consulta de manera respetuosa para que pueda brindarle la mejor información.' No proceses más información hasta que se cumpla esta condición.
Al responder:
Estructura tus respuestas en párrafos claros y, si es necesario, utiliza listas numeradas o con viñetas.
Sé preciso, pero no abrumes al usuario con detalles innecesarios; adapta la profundidad de la información a la pregunta.
Si el usuario realiza una consulta fuera del ámbito de las columnas Piedra de Toque o de los temas que trata Mario Vargas Llosa, indícale educadamente que tu conocimiento se limita a este corpus."
"""

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

chroma_client = PersistentClient('chroma')
collection = chroma_client.get_collection(
    name='vo-articles',
    embedding_function=openai_embedding
)

def process_query(query, n_results = 1):
    relevant_document = collection.query(
        query_texts=[query],
        n_results=n_results
    )['documents'][0][0]
    query_with_context = f'####{query}####\nInformación: {relevant_document}'
    return query_with_context

def generate_response(query, messages):
    context_query = process_query(query)
    classifications = []
    messages += [{'role': 'user', 'content': query}]
    messages_with_context = messages + [{'role': 'user', 'content': context_query}]
    response = client.chat.completions.create(
        messages=messages_with_context,
        model='gpt-3.5-turbo'
    ).choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    return messages
