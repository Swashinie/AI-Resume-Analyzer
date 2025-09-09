import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
    text = ' '.join(text.split())
    return text
