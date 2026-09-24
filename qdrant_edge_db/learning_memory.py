from pathlib import Path
from datetime import datetime, timezone

from helper import (
    new_shard,
    load_shard,
    seed_objects,
    object_photos,
    photo_uploader,
    teach,
    recognize,
    recognition_result,
    show_taught_photos,
    threshold_calibration,
    load_day_and_history,
    store_day,
    add_memory,
    embed_text,
    embed_image,
    save_shard,
    recent_text_search,
    text_search,
    photo_search,
    results_table,
    answers_table,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OBJECT_SHARD_DIR = BASE_DIR / "object_shard"
ASSISTANT_SHARD_DIR = BASE_DIR / "assistant_shard"

TEACH_DIR = BASE_DIR / "ro_shared_data" / "teach"
TEST_DIR = BASE_DIR / "ro_shared_data" / "test"

SUBJECT = "rubber duck"

THRESHOLD = 0.80

FRESHNESS_WEIGHT = 0.20

HALF_LIFE_DAYS = 7.0


# ============================================================
# START
# ============================================================

print()
print("=" * 76)
print("L5: LEARNING THROUGH MEMORY")
print("=" * 76)
print()


# ============================================================
# CHECK IMAGE DIRECTORIES
# ============================================================

print("Checking image directories...")
print()

print(f"Teach directory : {TEACH_DIR}")
print(f"Test directory  : {TEST_DIR}")
print()


if not TEACH_DIR.exists():
    raise FileNotFoundError(
        f"Teaching directory does not exist:\n{TEACH_DIR}\n\n"
        f"Create it and add at least two teaching images."
    )


if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory does not exist:\n{TEST_DIR}\n\n"
        f"Create it and add at least one test image."
    )


teach_photos = object_photos("teach")

test_photos = object_photos("test")


print(f"Teaching photos found : {len(teach_photos)}")
print(f"Test photos found     : {len(test_photos)}")
print()


if len(teach_photos) < 2:
    raise RuntimeError(
        "At least TWO teaching photos are required.\n\n"
        f"Put two images inside:\n{TEACH_DIR}"
    )


if len(test_photos) < 1:
    raise RuntimeError(
        "At least ONE test photo is required.\n\n"
        f"Put one image inside:\n{TEST_DIR}"
    )


# ============================================================
# CREATE OBJECT MEMORY SHARD
# ============================================================

print("=" * 76)
print("1. CREATE OBJECT MEMORY")
print("=" * 76)
print()

object_shard = new_shard(
    OBJECT_SHARD_DIR,
    image=512,
)

print(f"Object shard created: {OBJECT_SHARD_DIR}")
print()


# ============================================================
# SEED EXAMPLE OBJECTS
# ============================================================

print("Seeding example objects...")
print()

seed_objects(object_shard)

print()


# ============================================================
# SHOW EXAMPLE OBJECT LOCATIONS
# ============================================================

print("=" * 76)
print("2. EXAMPLE OBJECT LOCATIONS")
print("=" * 76)
print()

try:
    example_photos = object_photos("objects")

    if example_photos:
        print("Example object photos:")
        for photo in example_photos:
            print(f"  {photo}")
    else:
        print("No example object photos found.")

except Exception as exc:
    print(f"Could not show example objects: {exc}")

print()


# ============================================================
# SHOW PHOTO INSTRUCTIONS
# ============================================================

print("=" * 76)
print("3. TEACHING / TEST PHOTOS")
print("=" * 76)
print()

photo_uploader("teach")

print()

photo_uploader("test")

print()


# ============================================================
# SELECT TEST PHOTO
# ============================================================

test_photo = test_photos[0]

print("Test photo:")
print(test_photo)
print()


# ============================================================
# RECOGNIZE BEFORE TEACHING
# ============================================================

print("=" * 76)
print("4. RECOGNITION BEFORE TEACHING")
print("=" * 76)
print()

try:

    before = recognize(
        object_shard,
        test_photo,
    )

    print(
        recognition_result(
            test_photo,
            before,
        )
    )

except Exception as exc:

    print(
        "Recognition before teaching failed:"
    )

    print(exc)

print()


# ============================================================
# SHOW TEACHING PHOTOS
# ============================================================

print("=" * 76)
print("5. TEACHING PHOTOS")
print("=" * 76)
print()

show_taught_photos(
    teach_photos,
    SUBJECT,
)

print()


# ============================================================
# TEACH THE NEW OBJECT
# ============================================================

print("=" * 76)
print("6. TEACH NEW OBJECT")
print("=" * 76)
print()

print(f"Subject: {SUBJECT}")
print()

for photo in teach_photos:
    print(f"Teaching image: {photo}")

print()


teach(
    object_shard,
    SUBJECT,
    teach_photos,
    100,
)

print()

print("Teaching completed.")
print()


# ============================================================
# RECOGNIZE AFTER TEACHING
# ============================================================

print("=" * 76)
print("7. RECOGNITION AFTER TEACHING")
print("=" * 76)
print()

after = recognize(
    object_shard,
    test_photo,
)

print(
    recognition_result(
        test_photo,
        after,
    )
)

print()


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

print("=" * 76)
print("8. SIMILARITY THRESHOLD")
print("=" * 76)
print()

calibration = threshold_calibration(
    object_shard,
    teach_photos,
    selected=THRESHOLD,
    current=(
        SUBJECT,
        after.score,
    ),
)

print(calibration)
print()

print(
    f"Using threshold: {THRESHOLD}"
)

print()


# ============================================================
# DETERMINE WHETHER OBJECT IS KNOWN
# ============================================================

is_known = False

if after:

    try:

        best_score = float(
            after.score
        )

        is_known = (
            best_score >= THRESHOLD
        )

        print(
            f"Best similarity score: {best_score:.4f}"
        )

        print(
            f"Known object: {is_known}"
        )

    except Exception as exc:

        print(
            "Could not calculate known-object status:"
        )

        print(exc)

else:

    print(
        "No recognition result was returned."
    )


print()


# ============================================================
# CLOSE OBJECT SHARD
# ============================================================

print("=" * 76)
print("9. OBJECT MEMORY COMPLETE")
print("=" * 76)
print()

try:

    save_shard(
        object_shard
    )

except Exception as exc:

    print(
        f"Object shard save warning: {exc}"
    )


try:

    object_shard.close()

except Exception as exc:

    print(
        f"Object shard close warning: {exc}"
    )


print()


# ============================================================
# CREATE ASSISTANT MEMORY SHARD
# ============================================================

print("=" * 76)
print("10. CREATE ASSISTANT MEMORY")
print("=" * 76)
print()

assistant_shard = new_shard(
    ASSISTANT_SHARD_DIR,
    text=768,
    image=512,
)

print(
    f"Assistant shard created: "
    f"{ASSISTANT_SHARD_DIR}"
)

print()


# ============================================================
# LOAD DAY + HISTORY
# ============================================================

print("=" * 76)
print("11. LOAD DAY AND HISTORY")
print("=" * 76)
print()

day_data, notes, photos = load_day_and_history()
print("Day data loaded.")
print()

print("History loaded.")
print()


# ============================================================
# STORE DAY MEMORY
# ============================================================

print("=" * 76)
print("12. STORE DAY MEMORY")
print("=" * 76)
print()

try:

    store_day(
        assistant_shard,
        notes,
        photos,
    )

    print(
        "Day and history stored."
    )

except Exception as exc:

    print(
        "Could not store day/history:"
    )

    print(exc)

print()


# ============================================================
# ADD COMBINED OBJECT MEMORY
# ============================================================

print("=" * 76)
print("13. STORE LEARNED OBJECT MEMORY")
print("=" * 76)
print()

timestamp = datetime.now(
    timezone.utc
).isoformat()

note = (
    f"This is my {SUBJECT}. "
    f"I taught the assistant this object "
    f"using two photos."
)

print(f"Label     : {SUBJECT}")
print(f"Timestamp : {timestamp}")
print(f"Note      : {note}")
print()


# ------------------------------------------------------------
# Store text + image memory
# ------------------------------------------------------------

memory = add_memory(
    assistant_shard,
    memory_id=200,
    vectors={
        "text": embed_text(
            [note]
        )[0],
        "image": embed_image(
            [teach_photos[0]]
        )[0],
    },
    payload={
        "note": note,
        "subject": SUBJECT,
        "label": SUBJECT,
        "file": str(teach_photos[0]),
        "source_type": "learned_object",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    },
)
print(
    f"Combined memory stored: {memory.id}"
)

print()


# ============================================================
# SAVE / FLUSH ASSISTANT MEMORY
# ============================================================

print("=" * 76)
print("14. SAVE ASSISTANT MEMORY")
print("=" * 76)
print()

try:

    save_shard(
        assistant_shard
    )

    print(
        "Assistant shard saved."
    )

except Exception as exc:

    print(
        f"Save warning: {exc}"
    )

print()


# ============================================================
# MEANING-ONLY SEARCH
# ============================================================

print("=" * 76)
print("15. MEANING-ONLY SEARCH")
print("=" * 76)
print()

meaning_query = (
    "what is my gym locker code?"
)

print(
    f"Query: {meaning_query}"
)

print()

meaning_results = text_search(
    assistant_shard,
    meaning_query,
    limit=3,
)

if meaning_results:

    results_table(
        meaning_results,
        title="Meaning-only search",
        price=False,
    )

else:

    print(
        "No meaning-only results found."
    )

print()


# ============================================================
# FIND LATEST TIMESTAMP
# ============================================================

latest_timestamp = timestamp

print("=" * 76)
print("16. FRESHNESS RANKING")
print("=" * 76)
print()

print(
    f"Reference timestamp: "
    f"{latest_timestamp}"
)

print()


# ============================================================
# SEARCH WITH FRESHNESS
# ============================================================

by_freshness = recent_text_search(
    assistant_shard,
    meaning_query,
    latest_timestamp,
    freshness_weight=FRESHNESS_WEIGHT,
    half_life_days=HALF_LIFE_DAYS,
    limit=3,
)


# ============================================================
# DISPLAY QDRANT RESULTS
# ============================================================

if by_freshness:

    results_table(
        [
            item["hit"]
            for item in by_freshness
        ],
        title="Meaning + freshness",
        price=False,
    )

else:

    print(
        "No freshness-ranked results found."
    )


# ============================================================
# DISPLAY FRESHNESS SCORES
# ============================================================

print()

print(
    "Freshness ranking details:"
)

for item in by_freshness:

    print(
        f"original="
        f"{item['original_score']:.4f} | "
        f"freshness="
        f"{item['freshness_boost']:.4f} | "
        f"final="
        f"{item['adjusted_score']:.4f} | "
        f"age_days="
        f"{item['age_days']}"
    )

print()


# ============================================================
# SAVE AND CLOSE
# ============================================================

print("=" * 76)
print("17. CLOSE ASSISTANT SHARD")
print("=" * 76)
print()

try:

    save_shard(
        assistant_shard
    )

except Exception as exc:

    print(
        f"Save warning: {exc}"
    )


try:

    assistant_shard.close()

except Exception as exc:

    print(
        f"Close warning: {exc}"
    )

print(
    "Assistant shard closed."
)

print()


# ============================================================
# RELOAD ASSISTANT SHARD
# ============================================================

print("=" * 76)
print("18. RELOAD ASSISTANT MEMORY")
print("=" * 76)
print()

assistant_shard = load_shard(
    ASSISTANT_SHARD_DIR,
)

print(
    "Assistant shard reloaded."
)

print()


# ============================================================
# QUESTIONS
# ============================================================

questions = [
    "what did I get at the toy shop?",
    "show me the bakery",
    "where did I have brunch",
    "what did the bike chain cost",
]


# ============================================================
# ASK QUESTIONS
# ============================================================

print("=" * 76)
print("19. ASK MEMORY QUESTIONS")
print("=" * 76)
print()


for question in questions:

    print()
    print("-" * 76)
    print(
        f"QUESTION: {question}"
    )
    print("-" * 76)

    # --------------------------------------------------------
    # Text search
    # --------------------------------------------------------

    try:

        text_results = recent_text_search(
            assistant_shard,
            question,
            latest_timestamp,
            freshness_weight=FRESHNESS_WEIGHT,
            half_life_days=HALF_LIFE_DAYS,
            limit=3,
        )

    except Exception as exc:

        print(
            f"Text search failed: {exc}"
        )

        text_results = []


    # --------------------------------------------------------
    # Display text results
    # --------------------------------------------------------

    if text_results:

        print()

        results_table(
            [
                item["hit"]
                for item in text_results
            ],
            title="Text memory results",
            price=False,
        )

    else:

        print(
            "No text memory results."
        )


    # --------------------------------------------------------
    # Display freshness information
    # --------------------------------------------------------

    if text_results:

        print()

        print(
            "Freshness details:"
        )

        for item in text_results:

            print(
                f"score="
                f"{item['adjusted_score']:.4f} | "
                f"semantic="
                f"{item['original_score']:.4f} | "
                f"freshness="
                f"{item['freshness_boost']:.4f} | "
                f"age_days="
                f"{item['age_days']}"
            )


    # --------------------------------------------------------
    # Image search
    # --------------------------------------------------------

    try:

        image_results = photo_search(
            assistant_shard,
            question,
            limit=3,
        )

    except Exception as exc:

        print(
            f"Image search failed: {exc}"
        )

        image_results = []


    # --------------------------------------------------------
    # Display image results
    # --------------------------------------------------------

    if image_results:

        print()

        results_table(
            image_results,
            title="Image memory results",
            price=False,
        )

    else:

        print(
            "No image memory results."
        )


    print()


# ============================================================
# FINAL SAVE
# ============================================================

print("=" * 76)
print("20. FINAL SAVE")
print("=" * 76)
print()

try:

    save_shard(
        assistant_shard
    )

    print(
        "Final assistant memory saved."
    )

except Exception as exc:

    print(
        f"Final save warning: {exc}"
    )


# ============================================================
# FINAL CLOSE
# ============================================================

try:

    assistant_shard.close()

except Exception as exc:

    print(
        f"Final close warning: {exc}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 76)
print("LEARNING THROUGH MEMORY COMPLETE")
print("=" * 76)
print()
print(
    "The assistant learned the object:"
)
print(
    f"  {SUBJECT}"
)
print()
print(
    "It stored:"
)
print(
    "  - text memory"
)
print(
    "  - image memory"
)
print(
    "  - timestamp"
)
print(
    "  - object label"
)
print(
    "  - source information"
)
print()
print(
    "It then performed:"
)
print(
    "  - semantic search"
)
print(
    "  - freshness ranking"
)
print(
    "  - image search"
)
print(
    "  - shard reload"
)
print()
print("Done.")
