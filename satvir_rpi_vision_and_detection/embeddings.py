from fastembed import ImageEmbedding,TextEmbedding
import os
import numpy as np
from pathlib import Path
from functools import lru_cache
from config import IMAGE_EMBEDDING_MODEL,TEXT_EMBEDDING_MODEL,ENCODER_THREADS
CACHE_DIR = (os.environ.get("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed"))
from top_k_retrival import *

@lru_cache(maxsize=1)
def _textmodel():
    textmodel = TextEmbedding(TEXT_EMBEDDING_MODEL,threads=ENCODER_THREADS,cache_dir=CACHE_DIR)
    return textmodel
@lru_cache(maxsize=1)
def _imagemodel():
    imagemodel = ImageEmbedding(IMAGE_EMBEDDING_MODEL,threads=ENCODER_THREADS,cache_dir=CACHE_DIR)
    return imagemodel

def embedding_list_to_array(embeddings_list):
    embeddings_list2 = np.array(embeddings_list)
#     print(embeddings_list2.shape)
    return embeddings_list2.shape
def generate_text_to_vectorembedding(list_texts):
    textmodel=_textmodel()
    embeddings_generator = textmodel.embed(list_texts)
#     for doc, vector in zip(list_texts, embeddings_generator):
#        print("Document:", doc)
#        print(f"Vector of type: {type(vector)} with shape: {vector.shape}")
    embeddings_list = list(embeddings_generator)
    embedding_list_to_array(embeddings_list)
    return embeddings_list

def embed_text_textstring(text):
    textmodel=_textmodel()
    return next(textmodel.embed([text])).tolist()

def embed_query_textstring(text):
    textmodel=_textmodel()
    return next(textmodel.query_embed([text])).tolist()

def embed_image_crop_single(bgr):
    from PIL import Image
    img = Image.fromarray(bgr[:, :, ::-1])
    imagemodel=_imagemodel()
    return next(imagemodel.embed([img])).tolist()

def embed_image_singleimage(image_src):
    imagemodel=_imagemodel()
    return next(imagemodel.embed([image_src])).tolist()
def all_embeddings(all_lists):
    embedding_model =_textmodel()
    embeddings: list[np.ndarray] = list(embedding_model.passage_embed(all_lists))
    return embeddings



def generate_image_to_vectorembedding(list_images):
    imagemodel=_imagemodel()
    embeddings_generator = imagemodel.embed(list_images)
    embeddings_list = list(embeddings_generator)
    embedding_list_to_array(embeddings_list)
    return embeddings_list
def query_text_embedding(query_text_list):
    textmodel=_textmodel()
    embeddings_query=textmodel.query_embed(query_text_list)
    query_embeddings = list(embeddings_query)
    embedding_list_to_array(query_embeddings)
    return query_embeddings

def query_image_embedding(query_image):
    imagemodel=_imagemodel()
    embeddings_query=imagemodel.query_embed(query_image)
    query_embeddings = list(embeddings_query)
    return query_embeddings


# if __name__=='__main__':
#     list_images=["/home/satvir/build_ai_assistant_on_rpi/satvir_rpi_vision_and_detection/rubberduck_1.jpg"]
#     generate_image_to_vectorembedding(list_images)
#     list_texts = ["This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",]
#     generate_text_to_vectorembedding(list_texts)
#     list_texts = [
#         "ColBERT is a late interaction text embedding model, however, there are also other models such as TwinBERT.",
#         "On the contrary to the late interaction models, the early interaction models contains interaction steps at embedding generation process",
#         "dog name is alibaba",
#         "satvir is developer",
#         ]
#     queries = [
#         "what is my dog name",
#         "who is satvir",
#         "Are there any other late interaction text embedding models except ColBERT?",
#         "What is the difference between late interaction and early interaction text embedding models?",
#     ]
#     text_embeddings=generate_text_to_vectorembedding(list_texts)
#     query_embedding=query_text_embedding(queries)
#     sorted_indices=compute_relevance_scores_using_cosine(query_embedding, text_embeddings, k=1)
#     print("Sorted document indices:", sorted_indices)
#     for query_index, query in enumerate(queries):
#
#         print(f"\nQuery: {query}")
#
#         for document_index in sorted_indices[query_index]:
#             print(f"text: {list_texts[document_index]}")
#

