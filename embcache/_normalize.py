import re
import unicodedata

def normalize(query: str) -> str:
    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", query)
    # 2. Lowercase
    text = text.lower()
    # 3. Strip punctuation (Unicode category P* and S*)
    # We use a regex that matches any character where the category starts with P or S
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(('P', 'S')))
    # 4. Collapse all whitespace to single space
    text = re.sub(r"\s+", " ", text)
    # 5. Strip leading/trailing whitespace
    return text.strip()
