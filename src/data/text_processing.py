from pyvi.ViTokenizer import ViTokenizer


def remove_strip(text: str) -> str:
    text = text.strip(""" !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
    return text


def text_normalize(text: str) -> str:
    text = text.lower()
    return remove_strip(text)


def word_segment(text: str) -> str:
    return ViTokenizer.tokenize(text)
