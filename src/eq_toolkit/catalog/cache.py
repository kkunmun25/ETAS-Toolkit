from pathlib import Path

CACHE_DIR = Path.home() / ".etas_cache"

CACHE_DIR.mkdir(exist_ok=True)

def save_text(filename: str, text: str):
    """
    Save text into the cache.
    """
    filepath = CACHE_DIR / filename
    filepath.write_text(text, encoding="utf-8")

def load_text(filename: str) -> str:
    """
    Load text from the cache.
    """
    filepath = CACHE_DIR / filename
    return filepath.read_text(encoding="utf-8")

save_text("test.txt", "Hello ETAS!")

text = load_text("test.txt")

print(text)