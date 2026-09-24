from pathlib import Path
import json
import math
import shutil
from datetime import datetime, timezone

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    FieldCondition,
    Filter,
    MatchValue,
    Point,
    Query,
    QueryRequest,
    UpdateOperation,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"


# ============================================================
# FASTEMBED
# ============================================================

try:
    from fastembed import TextEmbedding
    from fastembed import ImageEmbedding

except ImportError:
    TextEmbedding = None
    ImageEmbedding = None


# ============================================================
# MODEL NAMES
# ============================================================

TEXT_MODEL_NAME = "BAAI/bge-base-en-v1.5"

CLIP_TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"

CLIP_IMAGE_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"


# ============================================================
# GLOBAL MODELS
# ============================================================

_text_model = None
_clip_text_model = None
_clip_image_model = None


# ============================================================
# MODEL LOADERS
# ============================================================

def get_text_model():

    global _text_model

    if _text_model is None:

        if TextEmbedding is None:
            raise ImportError(
                "fastembed is not installed.\n"
                "Run:\n"
                "uv add fastembed"
            )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("Loading text embedding model...")
        print(TEXT_MODEL_NAME)

        _text_model = TextEmbedding(
            model_name=TEXT_MODEL_NAME,
            cache_dir=str(MODELS_DIR),
        )

        print("Text model loaded.")

    return _text_model


def get_clip_text_model():

    global _clip_text_model

    if _clip_text_model is None:

        if TextEmbedding is None:
            raise ImportError(
                "fastembed is not installed."
            )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("Loading CLIP text model...")
        print(CLIP_TEXT_MODEL_NAME)

        _clip_text_model = TextEmbedding(
            model_name=CLIP_TEXT_MODEL_NAME,
            cache_dir=str(MODELS_DIR),
        )

        print("CLIP text model loaded.")

    return _clip_text_model


def get_clip_image_model():

    global _clip_image_model

    if ImageEmbedding is None:
        raise ImportError(
            "fastembed is not installed."
        )

    if _clip_image_model is None:

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("Loading CLIP image model...")
        print(CLIP_IMAGE_MODEL_NAME)

        _clip_image_model = ImageEmbedding(
            model_name=CLIP_IMAGE_MODEL_NAME,
            cache_dir=str(MODELS_DIR),
        )

        print("CLIP image model loaded.")

    return _clip_image_model


# ============================================================
# CREATE FRESH SHARD
# ============================================================

def new_shard(
    directory,
    text=None,
    image=None,
):

    directory = Path(directory)

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectors = {}

    if text is not None:

        vectors["text"] = EdgeVectorParams(
            size=text,
            distance=Distance.Cosine,
        )

    if image is not None:

        vectors["image"] = EdgeVectorParams(
            size=image,
            distance=Distance.Cosine,
        )

    if not vectors:
        raise ValueError(
            "At least one vector space is required."
        )

    config = EdgeConfig(
        vectors=vectors
    )

    shard = EdgeShard.create(
        str(directory),
        config,
    )

    print(
        "Created shard:",
        directory,
    )

    return shard


# ============================================================
# LOAD EXISTING SHARD
# ============================================================

def load_shard(directory):

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Shard directory not found: {directory}"
        )

    shard = EdgeShard.load(
        str(directory)
    )

    print(
        "Loaded shard:",
        directory,
    )

    return shard


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

def embed_text(texts):

    model = get_text_model()

    embeddings = model.embed(texts)

    return [
        vector.tolist()
        for vector in embeddings
    ]


def embed_query(text):

    return embed_text(
        [text]
    )[0]


# ============================================================
# IMAGE EMBEDDINGS
# ============================================================

def embed_image(paths):

    model = get_clip_image_model()

    image_paths = [
        Path(path)
        for path in paths
    ]

    embeddings = model.embed(
        image_paths
    )

    return [
        vector.tolist()
        for vector in embeddings
    ]


def embed_query_clip(text):

    model = get_clip_text_model()

    embeddings = model.embed(
        [text]
    )

    vector = next(
        iter(embeddings)
    )

    return vector.tolist()


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# ADD ONE MEMORY
# ============================================================

def add_memory(
    shard,
    memory_id,
    vectors,
    payload,
):

    memory = Point(
        id=memory_id,
        vector=vectors,
        payload=payload,
    )

    shard.update(
        UpdateOperation.upsert_points(
            [memory]
        )
    )

    return memory


# ============================================================
# TEACH OBJECT
# ============================================================

def teach(
    shard,
    subject,
    view_paths,
    start_id,
):

    if len(view_paths) == 0:

        raise ValueError(
            "No teaching photos supplied."
        )

    image_vectors = embed_image(
        view_paths
    )

    for i, (path, vector) in enumerate(
        zip(view_paths, image_vectors)
    ):

        add_memory(
            shard=shard,
            memory_id=start_id + i,
            vectors={
                "image": vector
            },
            payload={
                "label": subject,
                "file": str(path),
                "source_type": "object",
            },
        )

    shard.optimize()

    shard.flush()

    print()
    print(
        f"Taught '{subject}' using "
        f"{len(view_paths)} photos."
    )


# ============================================================
# RECOGNIZE IMAGE
# ============================================================

def recognize(
    shard,
    image_path,
):

    vector = embed_image(
        [image_path]
    )[0]

    results = shard.query(
        QueryRequest(
            query=Query.Nearest(
                vector,
                using="image",
            ),
            limit=1,
            with_payload=True,
        )
    )

    if not results:

        return None

    return results[0]


# ============================================================
# RECOGNITION RESULT
# ============================================================

def recognition_result(
    image_path,
    top_match,
    is_known=None,
    threshold=None,
):

    print()
    print("=" * 70)
    print("RECOGNITION RESULT")
    print("=" * 70)

    print(
        "Test image:",
        image_path,
    )

    if top_match is None:

        print(
            "No matching memory found."
        )

        print("=" * 70)

        return

    payload = (
        top_match.payload
        or {}
    )

    print(
        "Matched label:",
        payload.get(
            "label",
            "unknown",
        ),
    )

    print(
        "Matched file:",
        payload.get(
            "file",
            "unknown",
        ),
    )

    print(
        "Similarity score:",
        top_match.score,
    )

    if threshold is not None:

        print(
            "Threshold:",
            threshold,
        )

    if is_known is not None:

        print(
            "Known object:",
            is_known,
        )

    print("=" * 70)


# ============================================================
# DISPLAY TAUGHT PHOTOS
# ============================================================

def show_taught_photos(
    paths,
    subject,
):

    print()
    print("=" * 70)
    print(
        f"Photos taught as '{subject}'"
    )
    print("=" * 70)

    for index, path in enumerate(
        paths,
        start=1,
    ):

        print(
            f"{index}. {path}"
        )

    print("=" * 70)


# ============================================================
# TEXT SEARCH
# ============================================================

def text_search(
    shard,
    question,
    filter=None,
    limit=3,
):

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

    return shard.query(
        request
    )


# ============================================================
# IMAGE SEARCH
# ============================================================

def photo_search(
    shard,
    question,
    limit=3,
):

    query_vector = embed_query_clip(
        question
    )

    return shard.query(
        QueryRequest(
            query=Query.Nearest(
                query_vector,
                using="image",
            ),
            limit=limit,
            with_payload=True,
        )
    )


# ============================================================
# RESULTS TABLE
# ============================================================

def results_table(
    hits,
    title="Results",
    query=None,
    when=False,
    price=False,
    caption=None,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if query is not None:

        print(
            "Query:",
            query,
        )

        print("-" * 80)

    if not hits:

        print(
            "No results."
        )

        return hits

    for index, hit in enumerate(
        hits,
        start=1,
    ):

        payload = (
            hit.payload
            or {}
        )

        print(
            f"[{index}] "
            f"ID={hit.id} "
            f"score={hit.score:.4f}"
        )

        print(
            "    note:",
            payload.get(
                "note",
                "",
            ),
        )

        print(
            "    label:",
            payload.get(
                "label",
                "",
            ),
        )

        print(
            "    file:",
            payload.get(
                "file",
                "",
            ),
        )

        if when:

            print(
                "    timestamp:",
                payload.get(
                    "timestamp",
                    "",
                ),
            )

        if price:

            print(
                "    price:",
                payload.get(
                    "price",
                    "",
                ),
            )

    if caption:

        print()
        print(
            caption
        )

    print("=" * 80)

    return hits


# ============================================================
# SEED OBJECTS
# ============================================================

def seed_objects(
    shard,
):

    print()
    print(
        "Seeding example objects..."
    )

    object_dir = (
        BASE_DIR
        / "ro_shared_data"
        / "objects"
    )

    if not object_dir.exists():

        print(
            "Object directory does not exist:"
        )

        print(
            object_dir
        )

        print(
            "Skipping seed objects."
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
        for path in object_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in image_extensions
    ]

    if not image_files:

        print(
            "No seed object images found."
        )

        return

    image_vectors = embed_image(
        image_files
    )

    points = []

    for index, (
        path,
        vector,
    ) in enumerate(
        zip(
            image_files,
            image_vectors,
        ),
        start=1,
    ):

        point = Point(
            id=1000 + index,
            vector={
                "image": vector
            },
            payload={
                "label": path.stem,
                "file": str(path),
                "source_type": "seed_object",
            },
        )

        points.append(
            point
        )

    shard.update(
        UpdateOperation.upsert_points(
            points
        )
    )

    shard.optimize()

    shard.flush()

    print(
        "Seeded",
        len(points),
        "objects."
    )


# ============================================================
# PHOTO UPLOADER
# ============================================================

def photo_uploader(mode):

    print()
    print("=" * 70)

    if mode == "teach":

        print(
            "Teaching photos:"
        )

        print(
            "Place 2 or more photos in:"
        )

        print(
            BASE_DIR
            / "ro_shared_data"
            / "teach"
        )

    elif mode == "test":

        print(
            "Test photo:"
        )

        print(
            BASE_DIR
            / "ro_shared_data"
            / "test"
        )

    print("=" * 70)


# ============================================================
# OBJECT PHOTOS
# ============================================================

def object_photos(mode):

    if mode == "teach":

        directory = (
            BASE_DIR
            / "ro_shared_data"
            / "teach"
        )

    elif mode == "test":

        directory = (
            BASE_DIR
            / "ro_shared_data"
            / "test"
        )

    elif mode == "objects":

        directory = (
            BASE_DIR
            / "ro_shared_data"
            / "objects"
        )

    else:

        raise ValueError(
            "mode must be 'teach', 'test', or 'objects'"
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    files = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in extensions
        ]
    )

    if not files and mode == "objects":

        return []

    if not files:

        raise FileNotFoundError(
            f"No images found in {directory}"
        )

    return [
        str(path)
        for path in files
    ]


# ============================================================
# EXAMPLE OBJECT
# ============================================================

def show_example_object():

    print()
    print(
        "Example object:"
    )

    print(
        "Put your teaching photos in:"
    )

    print(
        BASE_DIR
        / "ro_shared_data"
        / "teach"
    )

    print()

    print(
        "Put your test photo in:"
    )

    print(
        BASE_DIR
        / "ro_shared_data"
        / "test"
    )


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def threshold_calibration(
    object_directory,
    image_directory,
    selected=0.80,
    current=None,
):

    print()
    print("=" * 70)
    print("THRESHOLD CALIBRATION")
    print("=" * 70)

    print(
        "Selected threshold:",
        selected,
    )

    print(
        "Example calibration:"
    )

    print(
        "Lowest known match: 0.86"
    )

    print(
        "Highest non-match: 0.74"
    )

    print(
        "Selected threshold:",
        selected,
    )

    if current is not None:

        subject, score = current

        print()
        print(
            "Current object:",
            subject,
        )

        print(
            "Current score:",
            score,
        )

        print(
            "Current decision:",
            score >= selected,
        )

    print("=" * 70)


# ============================================================
# FRESHNESS RANKING
# ============================================================

def freshness_ranking(
    latest_time,
    half_life,
    freshness_weight,
):

    return {
        "latest_time": latest_time,
        "half_life": half_life,
        "freshness_weight": freshness_weight,
    }


# ============================================================
# TIMESTAMP PARSER
# ============================================================

def parse_timestamp(value):

    if isinstance(
        value,
        (int, float),
    ):

        return float(value)

    if value is None:

        return 0.0

    text = str(value)

    try:

        return float(text)

    except ValueError:

        pass

    try:

        dt = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.timestamp()

    except ValueError:

        return 0.0


# ============================================================
# RECENCY SCORE
# ============================================================

def freshness_boost(
    timestamp,
    ranking,
):

    latest = ranking[
        "latest_time"
    ]

    half_life = ranking[
        "half_life"
    ]

    weight = ranking[
        "freshness_weight"
    ]

    timestamp_value = parse_timestamp(
        timestamp
    )

    latest_value = parse_timestamp(
        latest
    )

    age = max(
        0.0,
        latest_value - timestamp_value,
    )

    freshness = math.exp(
        -math.log(2)
        * age
        / half_life
    )

    return weight * freshness


# ============================================================
# RECENT TEXT SEARCH
# ============================================================
def recent_text_search(
    shard,
    query,
    latest_timestamp=None,
    *,
    freshness_weight=0.2,
    half_life_days=7.0,
    limit=5,
):
    """
    Semantic text search with freshness ranking.

    The Qdrant ScoredPoint objects are never modified.
    Freshness information is stored in normal Python dictionaries.
    """

    hits = text_search(
        shard,
        query,
        limit=limit,
    )

    # ---------------------------------------------------------
    # Reference timestamp
    # ---------------------------------------------------------

    if latest_timestamp is not None:
        reference_time = parse_timestamp(latest_timestamp)
    else:
        reference_time = datetime.now(timezone.utc).timestamp()

    if reference_time is None:
        reference_time = datetime.now(timezone.utc).timestamp()

    # ---------------------------------------------------------
    # Rank results
    # ---------------------------------------------------------

    ranked = []

    for hit in hits:

        payload = hit.payload or {}

        timestamp = payload.get("timestamp")

        if timestamp:

            dt = parse_timestamp(timestamp)

            if dt is not None:

                age_seconds = max(
                    0.0,
                    reference_time - dt,
                )

                age_days = age_seconds / 86400.0

                # -------------------------------------------------
                # Freshness calculation
                #
                # At age = 0:
                #     freshness = freshness_weight
                #
                # At age = half_life_days:
                #     freshness = freshness_weight / 2
                #
                # Older memories gradually receive less boost.
                # -------------------------------------------------

                freshness = 2 ** (
                    -age_days / half_life_days
                )

                boost = freshness_weight * freshness

            else:

                age_days = None
                boost = 0.0

        else:

            age_days = None
            boost = 0.0

        # ---------------------------------------------------------
        # Original semantic similarity
        # ---------------------------------------------------------

        original_score = float(hit.score)

        # ---------------------------------------------------------
        # Combined score
        # ---------------------------------------------------------

        adjusted_score = (
            original_score
            + boost
        )

        ranked.append(
            {
                "hit": hit,
                "original_score": original_score,
                "freshness_boost": boost,
                "adjusted_score": adjusted_score,
                "age_days": age_days,
            }
        )

    # ---------------------------------------------------------
    # Highest combined score first
    # ---------------------------------------------------------

    ranked.sort(
        key=lambda item: item["adjusted_score"],
        reverse=True,
    )

    return ranked[:limit]

# ============================================================
# LOAD DAY + HISTORY
# ============================================================

def load_day_and_history():

    data_file = (
        BASE_DIR
        / "ro_shared_data"
        / "day.json"
    )

    if data_file.exists():

        data = load_json(
            data_file
        )

        day = data.get(
            "day",
            [],
        )

        notes = data.get(
            "notes",
            [],
        )

        photos = data.get(
            "photos",
            [],
        )

        return (
            day,
            notes,
            photos,
        )

    # --------------------------------------------------------
    # Fallback example data
    # --------------------------------------------------------

    now = datetime.now(
        timezone.utc
    ).timestamp()

    day = [
        {
            "timestamp": now - 86400,
            "note": "I had brunch at the bakery.",
            "category": "food",
            "price": 12,
            "source_type": "text",
        },
        {
            "timestamp": now - 86400 * 3,
            "note": "My gym locker code is 4821.",
            "category": "personal",
            "price": 0,
            "source_type": "text",
        },
        {
            "timestamp": now - 86400 * 10,
            "note": "My gym locker code used to be 1397.",
            "category": "personal",
            "price": 0,
            "source_type": "text",
        },
        {
            "timestamp": now - 86400 * 2,
            "note": "I bought a bike chain for $18.",
            "category": "shopping",
            "price": 18,
            "source_type": "text",
        },
        {
            "timestamp": now - 86400,
            "note": "I got a rubber duck at the toy shop.",
            "category": "shopping",
            "price": 8,
            "source_type": "text",
        },
    ]

    notes = day

    photos = []

    return (
        day,
        notes,
        photos,
    )


# ============================================================
# STORE DAY
# ============================================================

def store_day(
    shard,
    notes,
    photos,
):

    print()
    print(
        "Storing day memories..."
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if notes:

        text_values = [
            item.get(
                "note",
                "",
            )
            for item in notes
        ]

        text_vectors = embed_text(
            text_values
        )

        points = []

        for index, (
            note,
            vector,
        ) in enumerate(
            zip(
                notes,
                text_vectors,
            ),
            start=2000,
        ):

            payload = dict(
                note
            )

            payload[
                "source_type"
            ] = "text"

            point = Point(
                id=index,
                vector={
                    "text": vector
                },
                payload=payload,
            )

            points.append(
                point
            )

        shard.update(
            UpdateOperation.upsert_points(
                points
            )

        )

        print(
            "Stored",
            len(points),
            "text memories."
        )

    # --------------------------------------------------------
    # PHOTOS
    # --------------------------------------------------------

    if photos:

        image_vectors = embed_image(
            photos
        )

        points = []

        for index, (
            photo,
            vector,
        ) in enumerate(
            zip(
                photos,
                image_vectors,
            ),
            start=3000,
        ):

            point = Point(
                id=index,
                vector={
                    "image": vector
                },
                payload={
                    "file": str(photo),
                    "source_type": "image",
                },
            )

            points.append(
                point
            )

        shard.update(
            UpdateOperation.upsert_points(
                points
            )
        )

        print(
            "Stored",
            len(points),
            "photo memories."
        )

    shard.optimize()
    shard.flush()


# ============================================================
# SAVE SHARD
# ============================================================

def save_shard(shard):

    shard.flush()

    print(
        "Shard flushed to disk."
    )


# ============================================================
# MEMORY RECEIPT
# ============================================================

def memory_receipt(
    shard,
    subject,
):

    print()
    print("=" * 70)
    print("MEMORY RECEIPT")
    print("=" * 70)

    print(
        "Subject:",
        subject,
    )

    try:

        print(
            "Points stored:",
            shard.info().points_count,
        )

    except Exception:

        print(
            "Unable to read point count."
        )

    print("=" * 70)


# ============================================================
# ANSWERS TABLE
# ============================================================

def answers_table(
    answers,
    image_directory,
):

    print()
    print("=" * 80)
    print("ASSISTANT ANSWERS")
    print("=" * 80)

    for index, (
        question,
        text_hit,
        image_hit,
    ) in enumerate(
        answers,
        start=1,
    ):

        print()
        print(
            f"[{index}] Question:"
        )

        print(
            question
        )

        print()

        if text_hit:

            print(
                "Text memory:"
            )

            print(
                "  score:",
                text_hit.score,
            )

            print(
                "  payload:",
                text_hit.payload,
            )

        else:

            print(
                "Text memory: none"
            )

        print()

        if image_hit:

            print(
                "Image memory:"
            )

            print(
                "  score:",
                image_hit.score,
            )

            print(
                "  payload:",
                image_hit.payload,
            )

        else:

            print(
                "Image memory: none"
            )

    print()
    print("=" * 80)
