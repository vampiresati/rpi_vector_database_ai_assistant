import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ENV_FILE = Path(__file__).resolve().parent / ".env"
if load_dotenv is not None:
    load_dotenv(ENV_FILE, override=False)

IMAGE_EMBEDDING_MODEL = "Qdrant/clip-ViT-B-32-vision"
IMAGE_EMBEDDING_DIM = 512
TEXT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
TEXT_EMBEDDING_DIM = 768
ENCODER_THREADS = 2
CACHE_DIR = (os.environ.get("FASTEMBED_CACHE_PATH")
             or str(Path.home() / ".cache" / "fastembed"))
WHISPER_MODEL = "whisper-base"
ASR_THREADS = 4


def _number(name, default, cast=float):
    """One calibration knob, with a legible error instead of a stack trace."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(
            f"{name} in {ENV_FILE.name} must be a number, got {raw!r}")


def _fractions(name, default):
    """Read one edge fraction or `left,top,right,bottom` fractions."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        parts *= 4
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit(
            f"{name} in {ENV_FILE.name} must be one number or four "
            f"(left,top,right,bottom), got {raw!r}")
    if len(vals) != 4 or not all(0 <= v < 0.5 for v in vals):
        raise SystemExit(
            f"{name} in {ENV_FILE.name} wants four fractions of 0 to 0.5 as "
            f"left,top,right,bottom, got {raw!r}")
    return vals


def _device(name):
    """An optional sounddevice name or numeric index."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw

RECOGNIZE_THRESHOLD = _number("RECOGNIZE_THRESHOLD", 0.90)
DETECT_CONF = _number("DETECT_CONF", 0.30)
DETECT_MAX_AREA = _number("DETECT_MAX_AREA", 0.20)
DETECT_MIN_AREA = _number("DETECT_MIN_AREA", 0.001)
CAMERA_ROTATE = _number("CAMERA_ROTATE", 0, int)
if CAMERA_ROTATE not in (0, 180):
    raise SystemExit(
        f"CAMERA_ROTATE in {ENV_FILE.name} must be 0 or 180, got "
        f"{CAMERA_ROTATE}. A quarter turn would swap the frame's width and "
        "height, which the rest of the app assumes is landscape.")


FRAME_CROP = _fractions("FRAME_CROP", (0.0, 0.0, 0.0, 0.0))
if FRAME_CROP[0] + FRAME_CROP[2] > 0.8 or FRAME_CROP[1] + FRAME_CROP[3] > 0.8:
    raise SystemExit(
        f"FRAME_CROP in {ENV_FILE.name} cuts away almost the whole frame: "
        f"{FRAME_CROP}")

MIC_DEVICE = _device("MIC_DEVICE")
