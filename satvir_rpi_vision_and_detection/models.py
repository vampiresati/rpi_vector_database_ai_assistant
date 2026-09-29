"""Flat-project model bridge for embeddings and voice transcription."""
import threading

import numpy as np

from embeddings import (
    embed_image_crop_single,
    embed_query_textstring,
    embed_text_textstring,
    _imagemodel,
    _textmodel,
)
from voice_related import _voice_model, transcribe


def embed_crop(bgr):
    return embed_image_crop_single(bgr)


def embed_text(text):
    return embed_text_textstring(text)


def embed_query(text):
    return embed_query_textstring(text)


_warm_lock = threading.Lock()


def warm_encoders():
    """Load speech and text models before a voice action takes the app lock."""
    with _warm_lock:
        _voice_model()
        _textmodel()


def warm_text():
    with _warm_lock:
        _textmodel()


def warm_up(progress=lambda name: None):
    progress("CLIP vision encoder")
    embed_crop(np.zeros((32, 32, 3), dtype=np.uint8))
