import streamlit as st
import pandas as pd
# import openai
from openai import OpenAI
import os
import openai

file_path = "chatbot_prueba.xlsx"
data = pd.read_excel(file_path)

# api_key = os.getenv("OPENAI_API_KEY")
api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = api_key

def generar_resumen(contenido):
    prompt = f"El contenido que te adjunto es un artículo de opinión de Mario Vargas Llosa en su columna Piedra de Toque. Lee el siguiente contenido, haz un resumen breve en menos de 100 palabras y muéstralo desde la perspectiva de Mario Vargas Llosa, enfocándote en su opinión: {contenido}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente que conoce la literatura y el recorrido de Mario Vargas Llosa, Premio Nobel de Literatura."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def generar_respuesta(pregunta, contenido):
    prompt = f"Basado en el siguiente contenido de un artículo de opinión de Mario Vargas Llosa, responde a la pregunta del usuario de manera concisa y clara en menos de 100 palabras: {contenido}. Pregunta: {pregunta}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en literatura y conocimiento general sobre Mario Vargas Llosa y sus artículos."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_articles" not in st.session_state:
    st.session_state.current_articles = pd.DataFrame()

if "mostrar_resumen" not in st.session_state:
    st.session_state.mostrar_resumen = True

if "selected_country" not in st.session_state:
    st.session_state.selected_country = None

st.title("Chatbot de Mario Vargas Llosa - Piedra de Toque")
st.write("¡Hola! Soy tu asistente para explorar los artículos de opinión de Mario Vargas Llosa en su columna 'Piedra de Toque'. Este asistente te mostrará la visión de Mario Vargas Llosa acerca de los temas que tú elijas.")


opcion = st.selectbox(
    "¿Por dónde desea empezar explorando?",
    ["Selecciona una opción", "1. Por temática", "2. Por país", "3. Por personaje", "4. Terminar"]
)

if opcion == "1. Por temática":
    st.write("Podemos dividir sus temáticas en 4 grandes ejes temáticos. Elija qué eje temático quisiera explorar:")
    tema = st.selectbox(
        "Selecciona un eje temático",
        ["Selecciona una opción", "1. Literatura y Crítica Cultural", "2. Historia y Memoria", 
         "3. Filosofía y Reflexión Intelectual", "4. Política y Sociedad", "5. Regresar al menú anterior"]
    )
    if tema != "Selecciona una opción":
        if tema == "5. Regresar al menú anterior":
            st.session_state.mostrar_resumen = True
            st.session_state.chat_history = []
        else:
            eje = tema.split(". ")[1]
            articulos = data[data["Eje temático"].str.contains(eje, na=False)]
            
            if not articulos.empty:
                st.write(f"Elegiste {eje}. Ahora, elige un país:")
                # paises = list(set(";".join(articulos["Países"].dropna()).split(";")))
                paises = list(set(";".join(articulos["Países"].fillna("Otros")).replace(" - ", ";").split(";")))
                
                # Seleccionar país
                # print(paises)
                # paises = ["Selecciona una opción"] + paises
                pais = st.selectbox("Selecciona un país", paises, key="country_selector")
                
                # Verificar si cambió el país seleccionado
                if st.session_state.selected_country != pais:
                    st.session_state.selected_country = pais
                    st.session_state.current_article = articulos[articulos["Países"].str.contains(pais, na=False)].iloc[0]
                    st.session_state.chat_history = []  # Limpiar historial
                    st.session_state.mostrar_resumen = True

                if st.session_state.current_article is not None:
                    article = st.session_state.current_article
                    if st.session_state.mostrar_resumen:
                        resumen = generar_resumen(article["Contenido"])
                        st.write(f"**Artículo:** {article['Título']}")
                        st.write(f"**Resumen:** {resumen}")
                        st.session_state.mostrar_resumen = False
                        st.write("---")

                    for msg in st.session_state.chat_history:
                        st.write(msg)

                    pregunta = st.text_input(f"¿Tienes alguna pregunta sobre este artículo? Escribe aquí (o escribe 'Salir' para terminar):", key="pregunta_input")
                    if pregunta:
                        if pregunta.lower() == "salir":
                            st.session_state.chat_history = []  # limpiar el historial
                            st.session_state.mostrar_resumen = True
                            st.write("Gracias por explorar los artículos. ¡Hasta pronto!")
                        else:
                            respuesta = generar_respuesta(pregunta, article["Contenido"])
                            st.session_state.chat_history.append(f"**Usuario:** {pregunta}")
                            st.session_state.chat_history.append(f"**Chatbot:** {respuesta}")
                            st.write(f"**Chatbot:** {respuesta}")