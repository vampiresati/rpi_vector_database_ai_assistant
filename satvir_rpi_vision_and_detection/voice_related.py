from functools import lru_cache
from config import WHISPER_MODEL,ASR_THREADS
LANGUAGE = "en"
def _sess_options():
    """ONNX Runtime's threading controls, for onnx-asr."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ASR_THREADS
    opts.inter_op_num_threads = ASR_THREADS
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts

@lru_cache(maxsize=1)
def _voice_model():
    import onnx_asr
    return onnx_asr.load_model(WHISPER_MODEL, sess_options=_sess_options(),providers=["CPUExecutionProvider"])

def transcribe(wav_path):
    """Local Whisper speech-to-text on one WAV file."""
    kwargs = {"language": LANGUAGE} if LANGUAGE else {}
    return _voice_model().recognize(wav_path, **kwargs).strip()



if __name__=='__main__':
    result=transcribe("/home/satvir/this_is_mobile_phone.wav")
    print(result)
    from voice_labels import norm_label,parse_label
    result=norm_label(result)
    print(result)
    result=parse_label(result)
    print(result)
#     result="This is mobile phone."
