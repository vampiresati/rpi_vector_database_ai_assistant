import threading
from voice_related import _voice_model
from embeddings import _textmodel,embed_image_crop_single
_warm_lock = threading.Lock()

def warm_encoders():
    with _warm_lock:
        _voice_model()      # every voice action transcribes first
        _textmodel()        # teach stores a text vector; ask queries one


def warm_text():
    """Load only the text encoder, using the shared model-loading lock."""
    with _warm_lock:
        _textmodel()


def warm_up(progress=lambda name: None):
    """Load the image encoder used by the camera loop."""
    progress("vision encoder")
    import numpy as np
    embed_image_crop_single(np.zeros((32, 32, 3), dtype=np.uint8))
