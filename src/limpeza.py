"""Funcoes de limpeza de texto dos anuncios (POC GLP-1 / Meta)."""
import html
import re

from spacy.lang.pt.stop_words import STOP_WORDS

# Palavras que NAO podem ser removidas (mudam o sentido do anuncio)
MANTER = {"não", "nem", "nunca", "sem", "mais", "menos", "grátis"}
STOPWORDS_PT = set(STOP_WORDS) - MANTER

# Textos de interface do Meta Ads (so os que nao carregam significado de venda)
TEXTOS_INTERFACE = [
    "patrocinado", "sponsored", "saiba mais", "saber mais",
    "learn more", "ver mais", "see more",
]

RE_TAG_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
RE_MENCAO = re.compile(r"@\w+")
RE_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF\u2600-\u27BF\uFE0F\u200D]"
)
RE_ESPACOS = re.compile(r"\s+")
RE_INTERFACE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in TEXTOS_INTERFACE) + r")\b",
    flags=re.IGNORECASE,
)


def contar_emojis(texto: str) -> int:
    """Conta emojis (guardamos como coluna, pode virar achado)."""
    return len(RE_EMOJI.findall(texto or ""))


def limpar_para_embedding(texto: str) -> str:
    """Limpeza leve: mantem frases completas, acentos e maiusculas."""
    if not isinstance(texto, str):
        return ""
    t = html.unescape(texto)
    t = RE_TAG_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENCAO.sub(" ", t)
    t = t.replace("#", " ")
    t = RE_INTERFACE.sub(" ", t)
    t = RE_EMOJI.sub(" ", t)
    return RE_ESPACOS.sub(" ", t).strip()


def limpar_para_vetorizacao(texto: str) -> str:
    """Limpeza forte (nuvens de palavras / c-TF-IDF): minusculas e sem stopwords."""
    t = limpar_para_embedding(texto).lower()
    t = re.sub(r"[^\w\s$%]", " ", t)
    palavras = [p for p in t.split() if p not in STOPWORDS_PT]
    return " ".join(palavras)