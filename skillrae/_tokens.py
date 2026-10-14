"""Token-budget utility extracted without benchmark orchestration."""
DEFAULT_TOKENIZER_NAME = "o200k_base"

def _load_tokenizer():
    try:
        import tiktoken
    except Exception:
        return None, "missing"

    try:
        return tiktoken.encoding_for_model("gpt-5.2"), "gpt-5.2"
    except Exception:
        return tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME), DEFAULT_TOKENIZER_NAME

def count_tokens(text: str) -> tuple[int, str]:
    tokenizer, tokenizer_name = _load_tokenizer()
    if tokenizer is None:
        return len(text.split()), "whitespace"
    return len(tokenizer.encode(text)), tokenizer_name
