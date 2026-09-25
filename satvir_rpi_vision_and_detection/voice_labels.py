"""Turning a spoken sentence into the object's name."""
import re

# ASR usually drops the punctuation that would end the naming clause, so the
# label also stops at words that start a new one ("...my mug I bought it").
_CLAUSE_WORDS = {"i", "it", "and", "that", "which", "because", "she", "he",
                 "they", "we", "you", "made", "bought", "got", "from", "at",
                 "about", "in", "on"}


def parse_label(transcript):
    """'This is my mug, Maria made it.' -> 'my mug'."""
    m = re.search(r"this is (?:an? )?(.+?)(?:\s*[,.;!?—–-]|$)",
                  transcript, re.IGNORECASE)
    phrase = m.group(1) if m else transcript
    words = []
    for w in phrase.split():
        if w.lower().strip(".,!?") in _CLAUSE_WORDS:
            break
        words.append(w)
    return " ".join(words[:5]).rstrip(".,!?").lower() or "unnamed"



def norm_label(text):
    return " ".join(text.split()).lower()

if __name__=="__main__":
    result=norm_label("this is my                          white          mug")
    print(result)
    result=parse_label(result)
    print(result)
