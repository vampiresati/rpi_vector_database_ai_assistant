from pathlib import Path
import json
import shutil
from qdrant_edge import (Query,QueryRequest,)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
TEXT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CLIP_TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
CLIP_IMAGE_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"

from fastembed import TextEmbedding,ImageEmbedding


_text_model = None
_clip_text_model = None
_clip_image_model = None


def fresh_start(directory):
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    return str(directory)



def get_text_model():
    """
    Load the text embedding model only once.
    """

    global _text_model

    if _text_model is None:

        if TextEmbedding is None:
            raise ImportError(
                "fastembed is not installed.\n"
                "Install it with:\n"
                "uv add fastembed"
            )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("Loading text embedding model...")
        print(
            "Model:",
            TEXT_MODEL_NAME,
        )

        _text_model = TextEmbedding(
            model_name=TEXT_MODEL_NAME,
            cache_dir=str(MODELS_DIR),
        )

        print("Text embedding model loaded.")

    return _text_model


# ============================================================
# TEXT DOCUMENT EMBEDDINGS
# ============================================================

def embed_text(texts):
    """
    Convert multiple documents into embeddings.

    Input:

        [
            "I like pizza",
            "I like coffee"
        ]

    Output:

        [
            [0.01, 0.02, ...],
            [0.03, 0.04, ...]
        ]
    """

    model = get_text_model()

    embeddings = model.embed(texts)

    vectors = []

    for embedding in embeddings:

        vectors.append(
            embedding.tolist()
        )

    return vectors


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(query):
    """
    Convert one search query into a vector.
    """

    vectors = embed_text(
        [query]
    )

    return vectors[0]


# ============================================================
# DISPLAY VECTOR INFORMATION
# ============================================================

def vector_preview(text, vector, count=10):
    """
    Display a small part of an embedding vector.
    """

    print()
    print("==========================================")
    print("VECTOR PREVIEW")
    print("==========================================")

    print("Text:")
    print(text)

    print()

    print(
        "Vector dimension:",
        len(vector),
    )

    print()

    print(
        f"First {count} values:"
    )

    print(
        vector[:count]
    )

    print("==========================================")


# ============================================================
# LOAD MEMORIES JSON
# ============================================================

def load_memories(
    filename,
    source_type=None,
):
    """
    Load memories from JSON.

    Expected format:

    [
        {
            "id": 1,
            "note": "Good pizza restaurant",
            "category": "food",
            "price": 12,
            "source_type": "text"
        }
    ]
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"Memory file not found: {filename}"
        )

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as file:

        memories = json.load(file)

    if not isinstance(memories, list):
        raise ValueError(
            "memories.json must contain a JSON list."
        )

    if source_type is not None:

        memories = [
            memory
            for memory in memories
            if memory.get("source_type") == source_type
        ]

    return memories


# ============================================================
# DISPLAY MEMORIES
# ============================================================

def memories_table(
    memories,
    title="Memories",
):
    """
    Simple terminal display for memories.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not memories:
        print("No memories found.")
        print("=" * 70)
        return

    for index, memory in enumerate(memories):

        print()
        print(
            f"[{index}] "
            f"id={memory.get('id')}"
        )

        print(
            "note:",
            memory.get("note", ""),
        )

        if "category" in memory:
            print(
                "category:",
                memory.get("category"),
            )

        if "price" in memory:
            print(
                "price:",
                memory.get("price"),
            )

        if "source_type" in memory:
            print(
                "source_type:",
                memory.get("source_type"),
            )

    print()
    print("=" * 70)


# ============================================================
# DISPLAY POINT
# ============================================================

def point_card(
    point,
    shown=0,
):
    """
    Display a Qdrant point.
    """

    print()
    print("=" * 70)
    print("POINT")
    print("=" * 70)

    print(
        "ID:",
        point.id,
    )

    print(
        "Payload:",
        point.payload,
    )

    if hasattr(point, "vector"):

        vector = point.vector

        if vector is not None:
            print(
                "Vector:",
                vector,
            )

    print(
        "Shown:",
        shown,
    )

    print("=" * 70)


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

def results_table(
    hits,
    title="Search Results",
    query=None,
    caption=None,
):
    """
    Display Qdrant search results.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if query is not None:

        print(
            "Query:",
            query,
        )

        print()

    if not hits:

        print("No results found.")

        if caption:
            print()
            print(caption)

        print("=" * 70)

        return hits

    for index, hit in enumerate(hits):

        print(
            f"[{index + 1}]"
        )

        print(
            "ID:",
            hit.id,
        )

        print(
            "Score:",
            hit.score,
        )

        print(
            "Payload:",
            hit.payload,
        )

        print("-" * 70)

    if caption:

        print()
        print(caption)

    print("=" * 70)

    return hits


# ============================================================
# TEXT SEARCH
# ============================================================
def text_search(
    shard,
    question,
    filter=None,
    limit=3,
):
    """
    Perform semantic text search.

    If filter is provided, it is passed directly
    into QueryRequest during construction.
    """

    question_vector = embed_query(
        question
    )

    if filter is None:

        request = QueryRequest(
            query=Query.Nearest(
                question_vector,
                using="text",
            ),
            limit=limit,
            with_payload=True,
        )

    else:

        request = QueryRequest(
            query=Query.Nearest(
                question_vector,
                using="text",
            ),
            filter=filter,
            limit=limit,
            with_payload=True,
        )

    hits = shard.query(
        request
    )

    return hits
# ============================================================
# CLIP TEXT MODEL
# ============================================================

def get_clip_text_model():
    """
    Load CLIP text embedding model.
    """

    global _clip_text_model

    if _clip_text_model is None:

        if TextEmbedding is None:
            raise ImportError(
                "fastembed is not installed.\n"
                "Install it with:\n"
                "uv add fastembed"
            )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "Loading CLIP text model..."
        )

        _clip_text_model = TextEmbedding(
            model_name=CLIP_TEXT_MODEL_NAME,
            cache_dir=str(MODELS_DIR),
        )

        print(
            "CLIP text model loaded."
        )

    return _clip_text_model


# ============================================================
# CLIP IMAGE MODEL
# ============================================================

def get_clip_image_model():
    """
    Load CLIP image embedding model.
    """

    global _clip_image_model

    if _clip_image_model is None:

        if ImageEmbedding is None:
            raise ImportError(
                "fastembed is not installed.\n"
                "Install it with:\n"
                "uv add fastembed"
            )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "Loading CLIP image model..."
        )

        _clip_image_model = ImageEmbedding(
            model_name=CLIP_IMAGE_MODEL_NAME,
            cache_dir=str(MODELS_DIR),
        )

        print(
            "CLIP image model loaded."
        )

    return _clip_image_model


# ============================================================
# CLIP TEXT QUERY EMBEDDING
# ============================================================

def embed_query_clip(
    text,
):
    """
    Convert a text description such as:

        "a red bicycle"

    into a 512-dimensional CLIP vector.
    """

    model = get_clip_text_model()

    embeddings = model.embed(
        [text]
    )

    embedding = next(
        iter(embeddings)
    )

    return embedding.tolist()


# ============================================================
# STORE PHOTOS
# ============================================================

def store_photos(
    shard,
    image_directory,
):
    """
    Read images from a directory,
    generate CLIP image embeddings,
    and store them in Qdrant Edge.

    Images are stored using the 'image'
    vector field.
    """

    from qdrant_edge import (
        Point,
        UpdateOperation,
    )

    image_directory = Path(
        image_directory
    )

    if not image_directory.exists():

        print(
            "Image directory does not exist:",
            image_directory,
        )

        return

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    image_files = [
        path
        for path in image_directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in image_extensions
    ]

    if not image_files:

        print(
            "No images found in:",
            image_directory,
        )

        return

    model = get_clip_image_model()

    print()
    print(
        "Found",
        len(image_files),
        "images.",
    )

    print(
        "Generating image embeddings..."
    )

    embeddings = model.embed(
        image_files
    )

    points = []

    for image_path, embedding in zip(
        image_files,
        embeddings,
    ):

        point = Point(
            id=image_path.stem,
            vector={
                "image": embedding.tolist()
            },
            payload={
                "source_type": "image",
                "filename": image_path.name,
                "path": str(image_path),
            },
        )

        points.append(point)

    shard.update(
        UpdateOperation.upsert_points(
            points
        )
    )

    print(
        "Stored",
        len(points),
        "image vectors.",
    )


# ============================================================
# DISPLAY IMAGE SEARCH RESULTS
# ============================================================

def show_photo_results(
    hits,
    image_directory,
    description,
):
    """
    Display image search results in the terminal.
    """

    image_directory = Path(
        image_directory
    )

    print()
    print("=" * 70)
    print("IMAGE SEARCH")
    print("=" * 70)

    print(
        "Description:",
        description,
    )

    if not hits:

        print(
            "No matching images found."
        )

        print("=" * 70)

        return

    for index, hit in enumerate(hits):

        payload = hit.payload or {}

        filename = payload.get(
            "filename"
        )

        if filename:

            image_path = (
                image_directory
                / filename
            )

        else:

            image_path = (
                image_directory
                / f"{hit.id}.jpg"
            )

        print()
        print(
            f"[{index + 1}]"
        )

        print(
            "ID:",
            hit.id,
        )

        print(
            "Score:",
            hit.score,
        )

        print(
            "Image:",
            image_path,
        )

        print(
            "Exists:",
            image_path.exists(),
        )

    print()
    print("=" * 70)
