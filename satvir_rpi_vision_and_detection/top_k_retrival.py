import numpy as np
def compute_relevance_scores(query_embedding, text_embeddings, k):
    query_embedding_arr=np.array(query_embedding)
    text_embeddings_arr=np.array(text_embeddings)
    text_embeddings_arrT=text_embeddings_arr.transpose(1,0)
    scores=np.matmul(query_embedding_arr, text_embeddings_arrT)
    sortingscores=np.argsort(scores, axis=1)
    reverseing_sorting=sortingscores[:, ::-1]
    top_k=reverseing_sorting[:, :k]
    sorted_indices =top_k[0]
    return sorted_indices

def compute_relevance_scores_using_cosine(query_embedding, text_embeddings, k):
    query_embedding_arr=np.array(query_embedding)
    text_embeddings_arr=np.array(text_embeddings)
    query_norm = np.linalg.norm(query_embedding_arr, axis=1, keepdims=True)
    text_norm = np.linalg.norm(text_embeddings_arr, axis=1, keepdims=True)
    query_normalized = query_embedding_arr / query_norm
    text_normalized = text_embeddings_arr / text_norm
    scores = np.matmul(query_normalized,text_normalized.T)
    sortingscores = np.argsort(scores, axis=1)
    reverseing_sorting = sortingscores[:, ::-1]
    top_k = reverseing_sorting[:, :k]
    sorted_indices = top_k[0]
    return top_k

def print_top_k(query_embedding, embeddings, documents, k=5):
    scores = np.dot(embeddings, query_embedding)
    sorted_scores = np.argsort(scores)[::-1]
    return sorted_scores
