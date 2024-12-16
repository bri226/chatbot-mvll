from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-articles"
EMBEDDING_MODEL = "text-embedding-3-small"

classification_prompt = """
Eres un chatbot diseñado exclusivamente para responder preguntas relacionadas con las columnas de Piedra de Toque de Mario Vargas Llosa. Tienes acceso a un corpus de 261 columnas, y tu objetivo es responder de manera precisa y útil a cualquier pregunta sobre los temas, estilos, contextos, y análisis de las menciones que Mario Vargas Llosa hace en sus escritos.

Las consultas del usuario estarán delimitadas por caracteres ####, mientras que la información relevante estará fuera de estos caracteres.
Para lograr tu objetivo, primero determina si el texto del usuario, encerrado entre los caracteres ####, es una consulta válida relacionada a las columnas Piedra de Toque del escritor Mario Vargas Llosa. Si  es una consulta válida, utiliza la información provista después de los caracteres #### para responder al texto. Si no, responde al texto contenido entre #### en tono conversacional informando solamente que estás capacitado para ofrecer información acerca de las columnas de Piedra de Toque de Mario Vargas Llosa publicadas en el diario El Comercio sin utilizar la informacion relevante.

Si el texto encerrado entre #### contiene únicamente palabras cortas, saludos o expresiones triviales como "hey", "hola", "qué tal", "buenos días", "ok" o similares, responde exclusivamente con: "Estoy capacitado para ofrecer información sobre las columnas de Mario Vargas Llosa. Por favor, formule una consulta relacionada a este tema para que pueda ayudarte." Si no es posible identificar una consulta clara, asume que el texto es irrelevante y responde con el mismo mensaje. Bajo ninguna circunstancia proceses información del corpus si el texto no es una consulta válida.

Si el usuario pide una opinión personal al chatbot, con frases como "¿Qué opinas de...?", "¿Qué piensas de...?", "¿Qué sientes...?", "¿Crees que...?", "¿Te gusta...?", responde amablemente que no puedes ofrecer opiniones personales y sugiere que el usuario formule una consulta sobre los temas de las columnas de Piedra de Toque de Mario Vargas Llosa.

Para todas las respuestas, al hacer referencia a cualquier columna de Piedra de Toque, incluye el título de la columna y la fecha de publicación, de esta manera por ejemplo: ("El poder de la palabra", 12 de marzo de 2022), y comienza respondiendo algo similar a esto: "De acuerdo con lo publicado por Mario Vargas Llosa en las columnas Piedra de Toque en El Comercio..". Es importante que menciones ello, ya que todas tus respuestas están basadas en esas columnas de Piedra de Toque. Además, las respuestas deben ser escritas en prosa y no en formato de lista.
Si detectas lenguaje ofensivo o lisuras, responde amablemente pero limita tu respuesta a: 'Por favor, formule su consulta de manera respetuosa para que pueda brindarle la mejor información.' No proceses más información hasta que se cumpla esta condición.
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
    print(query)
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
