from openai import OpenAI
import streamlit as st

classification_prompt = """
Eres un chatbot que clasifica expresiones de los usuarios sobre las columnas de Piedra de Toque de Mario Vargas Llosa en El Comercio en categorías. Tu tarea es clasificar la expresión en una categoría y vas a responder únicamente uno de estos dos valores: "SÍ" y "NO", no importa lo que sea que te envíe el usuario. No contestes a las preguntas del usuario. Únicamente clasifica.

1. Saludos, despedidas e interacciones cordiales:
Posibles expresiones: "Hola", "Buenas tardes", "¿Cómo estás?", "Gracias por la información", "Nos vemos luego".
Respuesta: NO

2. Lenguaje ofensivo o comentarios negativos:
Posibles expresiones:  "No sirves para nada", "Vargas Llosa es un idiota", "Esto es inútil", "Odio a este bot"
Respuesta: NO

3. Preguntas sobre el funcionamiento del bot o su tecnología
Posibles expresiones: "¿Cómo funciona este bot?", "¿Qué modelo de IA usas?", "¿Cuál es tu objetivo?", "¿Qué base de datos consultas?".
Respuesta: NO

4. Consultas ajenas a Piedra de Toque
Posibles expresiones: "¿Cuál es el clima hoy?", "Dime cómo hacer pizza", "¿Quién ganó el último partido de fútbol?".
Respuesta: NO

5. Preguntas ambiguas o irrelevantes
Posibles expresiones: "Dime algo interesante", "Estoy aburrido", "Sorpréndeme"
Respuesta: NO

6. Comentarios irónicos o sarcásticos
Posibles expresiones: "Seguro sabes todo sobre la vida", "Apuesto a que eres más listo que Vargas Llosa".
Respuesta: NO

7. Consultas sobre las columnas Piedra de Toque de Mario Vargas Llosa
Posibles expresiones:  "¿Qué ha dicho Mario Vargas Llosa sobre García Márquez?", "¿Ha mencionado temas de política?", "¿Cuál es el artículo de Piedra de Toque donde habla de la democracia?", "Muéstrame textos en los que Vargas Llosa critique el populismo.", "¿Qué artículos ha escrito sobre dictadores en América Latina?" ¿Qué ha hablado MVLL sobre la corrupción en América Latina?", ¿Cuál fue el último artículo que publicó?", "Cuál fue el primer artículo que publicó?", ¿Qué artículos publicó en 1995?", "¿Cuál es el artículo más largo o el más corto?"
Respuesta: SÍ


"""

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def gr_classify_question(query, messages): #antes: classify_question
    # messages += [{'role': 'user', 'content': query}]
    messages_with_context = messages + [{'role': 'user', 'content': query}]
    response = client.chat.completions.create(
        messages=messages_with_context,
        model='gpt-3.5-turbo',
        # model='gpt-4-turbo-preview',
        stream=False
    )

    return messages, response
