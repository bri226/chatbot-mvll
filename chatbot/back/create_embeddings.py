from uuid import uuid4
import tiktoken
from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

DATA_PATH = "./datos/docs"
METADATA_PATH = "./datos"
METADATA_FIELDS = {
        "AÑO": "AÑO",
        "FECHA": "FECHA",
        "TITULO": "TÍTULO",
        "NOMBRE_ARCHIVO": "NOMBRE_ARCHIVO"
}

load_dotenv()
tokenizer = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = "text-embedding-3-small"
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ENCODING_FORMAT = "Windows-1252"
metadata = pd.read_csv(f"{METADATA_PATH}/metadata.csv",sep=";",encoding=ENCODING_FORMAT)

def token_counter(text):
    """
    Calcula el número de tokens en un texto dado utilizando el tokenizer de tiktoken.
    Args:
        text (str): El texto que será tokenizado.
    Returns:
        int: La cantidad de tokens en el texto.
    Nota:
        Los tokens son las unidades mínimas de texto procesadas por modelos de lenguaje, 
        y su cantidad puede influir en los costos y limitaciones de los servicios de IA.
    """
    return len(tokenizer.encode(text))

# Configuración de un divisor de texto para dividir contenido en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", "\n", " "],  # Separadores utilizados para dividir el texto (prioridad descendente).
        chunk_size=8192,  # Tamaño máximo de cada fragmento en términos de tokens.
        chunk_overlap=100,  # Superposición de tokens entre fragmentos consecutivos para mantener contexto.
        length_function=token_counter  # Función personalizada para calcular la longitud del fragmento en tokens.
)

def document_to_text(document):
        """
        Extrae el texto de todas las páginas de un documento PDF.
        Args:
                document: Un objeto que representa el documento PDF, como el generado por PdfReader.
        Returns:
                str: El texto completo del documento, concatenado página por página.
        """
        document_text = ""  # Inicializa una cadena vacía para almacenar el texto del documento.
        for page in document.pages:  # Itera sobre todas las páginas del documento.
                document_text += page.extract_text()  # Extrae y concatena el texto de la página actual.
        return document_text  # Devuelve el texto completo extraído.

def entries_from_path(path):
        """
        Crea una entrada que contiene el texto extraído de un documento PDF y su metadata asociada.
        Args:
                path (str): Ruta al archivo PDF.
        Returns:
                dict: Un diccionario con las claves:
                - "text": Texto completo del documento.
                - "metadata": Diccionario con los campos de metadata correspondientes al documento.
        """
        entries = []
        print("Imprimiendo ruta: ", path)
        document_file_name = path.split("/")[-1]
        document = PdfReader(path)
        document_text = document_to_text(document)
        path_metadata_row = metadata[metadata.NOMBRE_ARCHIVO == document_file_name]
        
        print(f"Procesando archivo: {document_file_name}")
        print(f"Metadata encontrada: {path_metadata_row}")

        document_metadata = {
               key: str(path_metadata_row[METADATA_FIELDS[key]].values[0]) 
               for key in METADATA_FIELDS.keys()
        }
        # print({"text": document_text, "metadata": document_metadata})
        return {"text": document_text, "metadata": document_metadata}


def join_embeddings_chunks(chunks, embeddings):
        """
        Combina los fragmentos de texto (chunks) con sus embeddings y les asigna metadata y un identificador único.
        Args:
                chunks (list): Lista de objetos de texto divididos, que contienen tanto contenido como metadata.
                embeddings (list): Lista de vectores de embeddings generados para los chunks.
        Returns:
                list: Una lista de diccionarios, cada uno representando un embedding con su metadata y un identificador único.
        """

        print("Joining documents and embeddings...")
        chunks_as_dict = [chunk.metadata for chunk in chunks] # Extrae la metadata de cada chunk como un diccionario.
        
        # Agrega el texto de cada chunk como parte de su metadata.
        for chunk, metadata in zip(chunks, chunks_as_dict):
                metadata["text"] = chunk.page_content  # Incluye el contenido de texto del chunk en su metadata.
        
        # Combina cada embedding con su metadata y genera un ID único para cada entrada.
        embeddings_with_metadata = [
                {
                "values": embed,  # Los valores del vector de embedding.
                "metadata": chunk_metadata,  # La metadata asociada al chunk.
                "id": str(uuid4())  # Genera un identificador único para cada combinación.
                }
                for embed, chunk_metadata in zip(embeddings, chunks_as_dict)
        ]

        return embeddings_with_metadata  # Devuelve la lista de embeddings con metadata y ID.

def embeddings_from_chunks(chunks):
    """
    Genera embeddings para una lista de fragmentos de texto (chunks) y los combina con su metadata.
    Args:
        chunks (list): Lista de fragmentos de texto que incluyen el contenido (`page_content`) y metadata.
    Returns:
        list: Una lista de diccionarios, cada uno representando un embedding con su metadata y un identificador único.
    """

    print("Embedding Documents...")  # Mensaje para indicar que el proceso de creación de embeddings ha iniciado.

    # Llama a la API de OpenAI para generar embeddings basados en el contenido de los chunks.
    embeddings_response = openai_client.embeddings.create(
        input=[chunk.page_content for chunk in chunks],  # Extrae el contenido de texto de cada chunk.
        model=EMBEDDING_MODEL  # Especifica el modelo de embeddings a utilizar.
    )
    
    # Extrae los vectores de embeddings de la respuesta de OpenAI.
    embeddings = [entry.embedding for entry in embeddings_response.data]
    
    # Combina los embeddings generados con la metadata de los chunks.
    embeddings_with_metadata = join_embeddings_chunks(chunks, embeddings)
    
    return embeddings_with_metadata  # Devuelve la lista de embeddings con metadata y un identificador único.

def main():
    """
    Proceso principal que:
    1. Lee documentos PDF desde una ruta específica.
    2. Extrae su texto y metadata asociada.
    3. Divide los textos en fragmentos manejables (chunks).
    4. Genera embeddings para los fragmentos y los combina con su metadata.
    5. Guarda los embeddings generados en un archivo JSON.
    """
    # Obtiene las rutas de los documentos en la carpeta `DATA_PATH`.
    doc_paths = os.listdir(DATA_PATH)  # Lista todos los archivos en la carpeta especificada.
    doc_paths = [f"{DATA_PATH}/{doc}" for doc in doc_paths]  # Construye las rutas completas de los documentos.

    print("Reading documents...")  # Mensaje para indicar que el proceso de lectura de documentos ha iniciado.

    # Extrae texto y metadata de cada documento.
    corpus_texts = [entries_from_path(path) for path in doc_paths]
    
    # Divide el texto en fragmentos (chunks) con su metadata asociada.
    chunks = text_splitter.create_documents(
        texts=[entry["text"] for entry in corpus_texts],  # Lista de textos de los documentos.
        metadatas=[entry["metadata"] for entry in corpus_texts]  # Lista de metadatas correspondientes.
    )
    
    # Genera embeddings para los chunks y los combina con su metadata.
    embed_entries = embeddings_from_chunks(chunks)
    
    # Guarda los embeddings generados en un archivo JSON.
    with open("embeddings.json", "w", encoding=ENCODING_FORMAT) as f:
        json.dump(embed_entries, f)  # Serializa y escribe la lista de embeddings en el archivo.


# Protege la ejecución directa del script para permitir su importación en otros módulos sin ejecutarlo.
if __name__ == "__main__":
    main()
