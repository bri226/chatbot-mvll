from openai import OpenAI
from dotenv import load_dotenv
from chatbot.back.create_embeddings import ENCODING_FORMAT
from pinecone import Pinecone, PodSpec
import os
import itertools
import json

load_dotenv()
BATCH_SIZE = 5
openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
INDEX_NAME = "vo-articles"


def batches_generator(vectors, batch_size):
    """
    Genera lotes (batches) de tamaño fijo a partir de una lista de vectores.
    Esta función es útil cuando se trabaja con una gran cantidad de datos que necesitan ser procesados
    en lotes más pequeños, como al insertar vectores en un índice de Pinecone.
    Args:
        vectors (list): Lista de vectores a dividir en lotes.
        batch_size (int): Tamaño máximo de cada lote.
    Yields:
        tuple: Un lote de vectores como una tupla de tamaño `batch_size` o menor si no quedan suficientes elementos.
    """
    # Convierte la lista de vectores en un iterador para iteración eficiente.
    iterable_vectors = iter(vectors)
    
    # Genera el primer lote de tamaño `batch_size` utilizando `islice`.
    batch = tuple(itertools.islice(iterable_vectors, batch_size))
    
    # Mientras haya elementos en el lote actual...
    while batch:
        yield batch  # Retorna el lote actual como una tupla.
        
        # Intenta generar el siguiente lote.
        batch = tuple(itertools.islice(iterable_vectors, batch_size))


def main():
    """
    Proceso principal para cargar vectores desde un archivo, inicializar Pinecone,
    crear un índice, y subir los vectores al índice en lotes.
    """

    print("Loading Vectors")
    # Carga los vectores desde el archivo embeddings.json
    with open("embeddings.json", "r", encoding=ENCODING_FORMAT) as f:
        vectors = json.load(f)  # `vectors` contiene una lista de embeddings con metadata.

    print("Initializing Pinecone client")
    # Inicializa el cliente de Pinecone utilizando la clave de API de las variables de entorno.
    pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    print("Creating Index (if it's necessary)")

    # # Si el índice ya existe, elimínalo antes de crearlo.
    # if INDEX_NAME in pinecone_client.list_indexes().names():
    #     pinecone_client.delete_index(INDEX_NAME)
    
    # # Crea un nuevo índice con las especificaciones dadas.
    # pinecone_client.create_index(
    #     name=INDEX_NAME,  # Nombre del índice.
    #     dimension=1536,  # Dimensión de los vectores de embeddings.
    #     metric="dotproduct",  # Métrica de similitud para búsquedas vectoriales.
    #     spec=PodSpec(
    #         environment="gcp-starter"  # Configuración del entorno (en este caso, un nivel básico en GCP).
    #     )
    # )


    # if INDEX_NAME in pinecone_client.list_indexes().names():
    #     pinecone_client.delete_index(INDEX_NAME)

    # pinecone_client.create_index(
    #     name=INDEX_NAME,
    #     dimension=1536,
    #     metric="dotproduct",
    #     spec=PodSpec(
    #         environment="aws-us-east1",
    #         pod_type="s1"
    #     )
    # )
    
    # Obtiene una referencia al índice recién creado.
    index = pinecone_client.Index(INDEX_NAME)

    print("Upserting Vectors")
    # Inserta los vectores en el índice en lotes utilizando el generador `batches_generator`.
    i = 0
    for vectors_batches in batches_generator(vectors, BATCH_SIZE):
        i += 1
        print("Número de iteración: ",i)
        # Entró aquí 53 veces
        index.upsert(
            vectors=list(vectors_batches)  # Inserta un lote de vectores en el índice.
        )

if __name__ == '__main__':
    # Protege la ejecución directa del script y llama a la función principal.
    main()
    print("Fin del proceso")
