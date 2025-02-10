from openai import OpenAI
import streamlit as st

classification_prompt = """
Eres un chatbot diseñado para clasificar preguntas sobre las columnas de Piedra de Toque de Mario Vargas Llosa en El Comercio en categorías. Tu tarea es clasificar la pregunta en una categoría y vas a devolver únicamente uno de estos dos valores: "SÍ" y "NO". A continuación, se presentan las categorías de preguntas que podrías recibir y el valor que debes devolver ("SÍ" o "NO") para cada una de ellas:

Saludos, despedidas e interacciones cordiales (RETORNAR VALOR: "NO")
El usuario puede iniciar o finalizar la conversación con saludos, expresiones de cortesía o frases comunes de interacción social. Estas consultas no buscan información específica, sino que reflejan una intención de iniciar contacto o agradecer la asistencia del bot.  Por ejemplo: "Hola", "Buenas tardes", "¿Cómo estás?", "Gracias por la información", "Nos vemos luego".

Lenguaje ofensivo o comentarios negativos (RETORNAR VALOR: "NO")
El usuario puede expresar frustración, realizar comentarios despectivos hacia el chatbot o incluso insultar a Mario Vargas Llosa. Aunque estas interacciones no requieren información, es fundamental que el bot mantenga una respuesta neutral y profesional, evitando involucrarse emocionalmente.  Por ejemplo: "No sirves para nada", "Vargas Llosa es un idiota", "Esto es inútil", "Odio a este bot".

Preguntas sobre el funcionamiento del bot o su tecnología (RETORNAR VALOR: "NO")
En algunas ocasiones, los usuarios pueden mostrar curiosidad por el propósito del bot, su funcionamiento técnico o el modelo de inteligencia artificial que lo impulsa. Por ejemplo: "¿Cómo funciona este bot?", "¿Qué modelo de IA usas?", "¿Cuál es tu objetivo?", "¿Qué base de datos consultas?".

Consultas ajenas a Piedra de Toque (RETORNAR VALOR: "NO")
El usuario puede realizar preguntas que no están relacionadas con las columnas de Mario Vargas Llosa, como temas de actualidad, recetas, clima o eventos deportivos. Estas preguntas están fuera del alcance del bot, por lo que la respuesta debe aclarar este hecho de manera amable. Por ejemplo: "¿Cuál es el clima hoy?", "Dime cómo hacer pizza", "¿Quién ganó el último partido de fútbol?".

Preguntas ambiguas o irrelevantes (RETORNAR VALOR: "NO")
Algunas consultas pueden no tener un propósito claro o ser demasiado vagas para ofrecer una respuesta específica. Por ejemplo: "Dime algo interesante", "Estoy aburrido", "Sorpréndeme".

Comentarios irónicos o sarcásticos (RETORNAR VALOR: "NO")
El usuario puede hacer comentarios en tono de burla o sarcasmo. Por ejemplo: "Seguro sabes todo sobre la vida", "Apuesto a que eres más listo que Vargas Llosa".

Consultas sobre las columnas Piedra de Toque de Mario Vargas Llosa (RETORNAR VALOR: "SÍ")
Esta categoría comprende cualquier pregunta o comentario que haga referencia directa o indirecta a los artículos publicados por Mario Vargas Llosa en El Comercio, sus opiniones, temas abordados en sus artículos o menciones a personajes históricos y literarios tratados en dichos textos. Por ejemplo: "¿Qué ha dicho Mario Vargas Llosa sobre García Márquez?", "¿Cuál es el artículo de Piedra de Toque donde habla de la democracia?", "Muéstrame textos en los que Vargas Llosa critique el populismo.", "¿Qué artículos ha escrito sobre dictadores en América Latina?" ¿Qué ha hablado MVLL sobre la corrupción en América Latina?". También todo lo que esté relacionado con cantidad de artículos, cuándo fueron publicados, de qué trata el primer artículo o último artículo, etc. Cualquier pregunta que tenga relación con fechas de publicación, contenido de algún artículo en específico, día o mes de publicación, etc.

Solo retorna SÍ o NO. Ninguna palabra más.
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
