from top_k_retrival import print_top_k
from embeddings import embed_text_textstring,all_embeddings,query_image_embedding
if __name__ == "__main__":
    vector1 = embed_text_textstring("hi i am satvir")
    print("hi")

    vectr = embed_query_textstring("satvir")
#     print(vectr)
    list_texts = [
            "ColBERT is a late interaction text embedding model, however, there are also other models such as TwinBERT.",
            "On the contrary to the late interaction models, the early interaction models contains interaction steps at embedding generation process",
            "dog name is alibaba",
            "satvir is developer",
            ]
    embeddings=all_embeddings(list_texts)
    print(embeddings[0].shape, len(embeddings))
    query="who is satvir"
    print('query',query)
    query_embedding = embed_text_textstring(query)
    k=3
    sorted_scores=print_top_k(query_embedding, embeddings, k)
    for i in range(k):
        print(f"Rank {i+1}: {list_texts[sorted_scores[i]]}")
    query_image_embedding("/home/satvir/build_ai_assistant_on_rpi/satvir_rpi_vision_and_detection/rubberduck_1.jpg")
