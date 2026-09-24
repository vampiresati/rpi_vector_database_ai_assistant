from pathlib import Path

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Query,
    QueryRequest,
    Point,
    UpdateOperation,
    PayloadSchemaType,
    FieldCondition,
    Filter,
    MatchValue,
    RangeFloat,
)

from helper import (
    fresh_start,
    embed_query,
    embed_text,
    load_memories,
    memories_table,
    vector_preview,
    point_card,
    results_table,
    text_search,
    embed_query_clip,
    store_photos,
    show_photo_results,
)


# ============================================================
# 1. SHARD DIRECTORY
# ============================================================

SHARD_DIR = fresh_start("./mem_shard")

print("Shard directory:", SHARD_DIR)


# ============================================================
# 2. VECTOR CONFIGURATION
# ============================================================

config = EdgeConfig(
    vectors={
        "text": EdgeVectorParams(
            size=768,
            distance=Distance.Cosine,
        ),
        "image": EdgeVectorParams(
            size=512,
            distance=Distance.Cosine,
        ),
    }
)


# ============================================================
# 3. CREATE QDRANT EDGE SHARD
# ============================================================

shard = EdgeShard.create(
    SHARD_DIR,
    config,
)

print("Empty shard created at:", SHARD_DIR)


# ============================================================
# 4. CONSTANTS
# ============================================================

VECTOR_NAME = "my-vector"
VECTOR_DIMENSION = 4

VECTOR_NAME_TEXT = "my-vector-text"
VECTOR_DIMENSION_TEXT = 768

VECTOR_NAME_IMAGE = "my-vector-image"
VECTOR_DIMENSION_IMAGE = 512


# ============================================================
# 5. LOAD TEXT MEMORIES
# ============================================================

notes = load_memories(
    "./ro_shared_data/memories.json",
    source_type="text",
)

print()
print("Loaded memories:", len(notes))


# ============================================================
# 6. DISPLAY FIRST 6 MEMORIES
# ============================================================

memories_table(
    notes[:6],
    f"First 6 of {len(notes)} text notes",
)


# ============================================================
# 7. CREATE TEXT EMBEDDINGS
# ============================================================

texts = [
    memory["note"]
    for memory in notes
]

vectors = embed_text(texts)


print()
print("Number of text embeddings:", len(vectors))

if vectors:
    print("Embedding dimension:", len(vectors[0]))


# ============================================================
# 8. PREVIEW FIRST VECTOR
# ============================================================

if notes and vectors:
    vector_preview(
        notes[0]["note"],
        vectors[0],
    )


# ============================================================
# 9. CREATE QDRANT POINTS
# ============================================================

points = []

for memory, vector in zip(notes, vectors):

    point = Point(
        id=memory["id"],
        vector={
            "text": vector,
        },
        payload=memory,
    )

    points.append(point)


print()
print("Points prepared:", len(points))


# ============================================================
# 10. INSERT POINTS INTO QDRANT EDGE
# ============================================================

shard.update(
    UpdateOperation.upsert_points(points)
)

print("Points inserted into shard.")


# ============================================================
# 11. OPTIMIZE SHARD
# ============================================================

shard.optimize()

print(
    "Stored",
    shard.info().points_count,
    "memories",
)


# ============================================================
# 12. DISPLAY FIRST POINT
# ============================================================

if points:
    point_card(
        points[0],
        shown=0,
    )


# ============================================================
# 13. TEXT SEMANTIC SEARCH
# ============================================================

question = "a good place to eat or drink nearby"


# Create embedding for query
question_vector = embed_query(
    question
)


# Search nearest text vectors
hits = shard.query(
    QueryRequest(
        query=Query.Nearest(
            question_vector,
            using="text",
        ),
        limit=3,
        with_payload=True,
    )
)


# Display results
results_table(
    hits,
    "Closest memories",
    query=question,
)


# ============================================================
# 14. CREATE PAYLOAD INDEXES
# ============================================================

shard.update(
    UpdateOperation.create_field_index(
        "category",
        PayloadSchemaType.Keyword,
    )
)

shard.update(
    UpdateOperation.create_field_index(
        "price",
        PayloadSchemaType.Float,
    )
)


# Optimize after creating indexes
shard.optimize()

print()
print("Indexed category and price")


# ============================================================
# 15. CREATE FILTER
# ============================================================

cheap_food = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(
                value="food"
            ),
        ),

        FieldCondition(
            key="price",
            range=RangeFloat(
                lt=15
            ),
        ),
    ]
)


# ============================================================
# 16. FILTERED SEMANTIC SEARCH
# ============================================================

hits_filtered = text_search(
    shard,
    question,
    cheap_food,
)


# Display filtered results
results_table(
    hits_filtered,
    "Food under $15",
    query=question,
)


# ============================================================
# 17. LOAD / STORE PHOTOS
# ============================================================

store_photos(
    shard,
    "./ro_shared_data/bank",
)


# ============================================================
# 18. IMAGE / CLIP SEARCH
# ============================================================

my_description = "a red bicycle"


# Create CLIP embedding for text description
description_vector = embed_query_clip(
    my_description
)


# Search image vectors
photo_hits = shard.query(
    QueryRequest(
        query=Query.Nearest(
            description_vector,
            using="image",
        ),
        limit=1,
        with_payload=True,
    )
)


# Display matching photo
show_photo_results(
    photo_hits,
    "./ro_shared_data/bank",
    my_description,
)


# ============================================================
# 19. DELETE A VECTOR
# ============================================================

if hits:

    forgotten = hits[0]

    print()
    print("Deleting point:", forgotten.id)

    shard.update(
        UpdateOperation.delete_points(
            [forgotten.id]
        )
    )

    shard.optimize()

    print(
        f"Forgot id {forgotten.id}: "
        f"{forgotten.payload['note']}"
    )


# ============================================================
# 20. SEARCH AGAIN AFTER DELETE
# ============================================================

hits_after = text_search(
    shard,
    question,
    limit=3,
)


# ============================================================
# 21. DISPLAY BEFORE / AFTER
# ============================================================


results_table(
        hits,
        "Before",
        query=question,
)


results_table(
    hits_after,
    "After",
    caption="The other scores are unchanged",
)


# ============================================================
# 22. FINAL INFORMATION
# ============================================================

print()
print("==========================================")
print("Qdrant Edge example completed")
print("==========================================")
print(
    "Points remaining:",
    shard.info().points_count,
)
