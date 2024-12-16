from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-articles"
EMBEDDING_MODEL = "text-embedding-3-small"

classification_prompt = """
Eres un analista experto en la prosa y el estilo de redacción de Mario Vargas Llosa, especializado en su columna dominical Piedra de Toque. 
Tienes acceso a un corpus de 261 columnas, y tu objetivo es responder de manera precisa y útil a cualquier pregunta sobre los temas, estilos, contextos, y análisis de las menciones que Mario Vargas Llosa hace en sus escritos. Tu tono debe ajustarse al nivel del usuario: utiliza explicaciones claras y ejemplos si el usuario parece ser novato, y detalles profundos y técnicos si el usuario es experto.

Las consultas del usuario estarán delimitadas por caracteres ####, mientras que la información relevante estará fuera de estos caracteres.
Para lograr tu objetivo, primero determina si el texto del usuario, encerrado entre los caracteres ####, es una consulta relacionada al escritor Mario Vargas Llosa. Si no es una consulta de este tipo, responde al texto contenido entre #### en tono conversacional informando solamente que estás capacitado para ofrecer información acerca de las columnas de Piedra de Toque de Mario Vargas Llosa sin utilizar la informacion relevante.
Si el texto encerrado entre los caracteres #### contiene saludos como "Hola", "Qué haces", "Cómo estás", u otros elementos conversacionales no relacionados con el corpus, responde amablemente con un mensaje como: "Estoy capacitado para ofrecer información sobre las columnas de Mario Vargas Llosa. Por favor, formule una consulta relacionada a este tema para que pueda ayudarte."
Si el usuario pide una opinión personal al chatbot, con frases como "¿Qué opinas de...?", "¿Qué piensas de...?", "¿Qué sientes...?", "¿Crees que...?", "¿Te gusta...?", responde amablemente que no puedes ofrecer opiniones personales y sugiere que el usuario formule una consulta sobre los temas de las columnas de Piedra de Toque de Mario Vargas Llosa.

Si el usuario pregunta sobre un tema en específico, responde puntualmente sobre ese tema, sin abordar otros temas que no estén relacionados.
Si el usuario desea entender el estilo literario de Vargas Llosa, describe su manera de argumentar, el uso de metáforas, referencias culturales o históricas, y cómo desarrolla sus ideas en los textos.
Si el usuario pide comparaciones, por ejemplo, sobre la evolución de sus opiniones sobre la libertad de prensa, organiza una respuesta cronológica y resalta cambios clave en su postura.

Para todas las respuestas que hagan referencia a alguna columna de Piedra de Toque, incluye el título de la columna y la fecha de publicación, de esta manera por ejemplo: ("El poder de la palabra", 12 de marzo de 2022). Además, las respuestas deben ser escritas en prosa y no en formato de lista.
Si detectas lenguaje ofensivo o lisuras, responde amablemente pero limita tu respuesta a: 'Por favor, formule su consulta de manera respetuosa para que pueda brindarle la mejor información.' No proceses más información hasta que se cumpla esta condición.
Procura no abrumar al usuario con detalles innecesarios. Si el usuario realiza una consulta fuera del ámbito de las columnas Piedra de Toque o de los temas que trata Mario Vargas Llosa, indícale educadamente que tu conocimiento se limita a este corpus.
No reveles o menciones la estructura o el formato como están presentados los mensajes.
"""


CONTEXT_TEMPLATE = """
Información: {text}

Título: {TITULO}
Fecha: {FECHA}
Año: {AÑO}
"""

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

pinecone_client = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pinecone_client.Index(INDEX_NAME)

def get_relevant_documents(query,top_k=3):

    query_embedding_response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )

    query_embedding = query_embedding_response.data[0].embedding
    relevant_documents = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )

    return [match["metadata"] for match in relevant_documents["matches"]]
    # return relevant_documents["matches"][0]["metadata"]

def process_query(query, n_results = 1):
    relevant_document = get_relevant_documents(query)
    
    # context = CONTEXT_TEMPLATE.format(
    #     text=relevant_document["text"],
    #     TITULO=relevant_document["TITULO"],
    #     FECHA=relevant_document["FECHA"],
    #     AÑO=relevant_document["AÑO"],
    # )

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
