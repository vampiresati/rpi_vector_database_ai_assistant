from top_k_retrival import print_top_k
from warm_up import warm_encoders, warm_text, warm_up
from embeddings import embed_text_textstring,embed_query_textstring,all_embeddings,embed_image_singleimage,embed_image_crop_single
from voice_related import transcribe
from voice_labels import norm_label,parse_label


if __name__ == "__main__":
    print('hi')
    warm_up(print)
    warm_text()
    warm_encoders()
    vector1 = embed_text_textstring("hi i am satvir")
    vectr = embed_query_textstring("satvir")
    print(vectr[:3])
    print(vector1[:3])
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
    query_embedding = embed_query_textstring(query)
    k=3
    sorted_scores=print_top_k(query_embedding, embeddings, k)
    for i in range(k):
        print(f"Rank {i+1}: {list_texts[sorted_scores[i]]}")
    v=embed_image_singleimage("/home/satvir/build_ai_assistant_on_rpi/satvir_rpi_vision_and_detection/rubberduck_1.jpg")
    print(v[:5])
    result=transcribe("/home/satvir/this_is_mobile_phone.wav")
    print(result)
    result=norm_label(result)
    print(result)
    result=parse_label(result)
    print(result)
