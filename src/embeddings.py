"""
Embeddings dos anuncios (texto -> vetor de numeros que representa o significado).

- Calcula uma vez e SALVA EM DISCO (.npy + .json). Na proxima execucao, se os
  textos e o modelo forem os mesmos, carrega do disco em vez de recalcular.
- O arquivo .json guarda uma "impressao digital" (hash) dos textos e do modelo.
  Se voce mudar um texto, a ordem das linhas ou o modelo, o cache e ignorado
  automaticamente (evita usar embeddings desalinhados dos anuncios).

Modo de TESTE sem internet / sem baixar modelo:
    modelo="tfidf-teste"  -> usa TF-IDF + SVD. SO serve para testar o pipeline.
    NUNCA use esses vetores nos resultados do relatorio.

Uso pelo terminal (na raiz do projeto):
    python -m src.embeddings                 # modelo real (config.MODELO_EMBEDDING)
    python -m src.embeddings --tfidf         # teste rapido, sem baixar modelo
    python -m src.embeddings --dedup         # so um anuncio por grupo de copias
    python -m src.embeddings --forcar        # ignora o cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src import config
from src.cli import executar  # noqa: E402

MODELO_TESTE_TFIDF = "tfidf-teste"


def _impressao_digital(textos: list[str], modelo: str) -> str:
    h = hashlib.sha256()
    h.update(modelo.encode("utf-8"))
    for t in textos:                      # a ordem das linhas tambem conta
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def _caminhos(caminho_cache: str | Path | None) -> tuple[Path, Path]:
    npy = Path(caminho_cache or config.ARQ_EMBEDDINGS)
    return npy, npy.with_suffix(".json")


def carregar_embeddings(caminho_cache: str | Path | None = None) -> np.ndarray:
    """Le embeddings ja salvos (sem checar se combinam com os textos)."""
    npy, _ = _caminhos(caminho_cache)
    if not npy.exists():
        raise FileNotFoundError(f"Embeddings nao encontrados em {npy}. Rode: python -m src.embeddings")
    return np.load(npy)


def _embeddings_tfidf(textos: list[str], dimensoes: int = 100) -> np.ndarray:
    """Vetores de TESTE (sem rede neural). Nao usar nos resultados."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    matriz = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(textos)
    dimensoes = max(2, min(dimensoes, matriz.shape[0] - 1, matriz.shape[1] - 1))
    svd = TruncatedSVD(n_components=dimensoes, random_state=config.SEED)
    return normalize(svd.fit_transform(matriz)).astype(np.float32)


def _embeddings_modelo(textos: list[str], modelo: str, batch_size: int, verbose: bool) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as erro:
        raise ImportError(
            "O pacote 'sentence-transformers' nao esta instalado. "
            "Rode: pip install sentence-transformers  "
            "(ou use --tfidf para um teste sem modelo)") from erro
    if verbose:
        print(f"Carregando o modelo '{modelo}' (na 1a vez ele e baixado, ~1 GB)...")
    st = SentenceTransformer(modelo)
    return st.encode(textos, batch_size=batch_size, show_progress_bar=verbose,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def gerar_embeddings(textos: list[str], modelo: str | None = None,
                     caminho_cache: str | Path | None = None, batch_size: int = 32,
                     forcar: bool = False, verbose: bool = True) -> np.ndarray:
    """
    Devolve uma matriz (n_textos x dimensoes), na mesma ordem dos textos.
    Usa o cache em disco quando textos e modelo nao mudaram.
    """
    modelo = modelo or config.MODELO_EMBEDDING
    textos = ["" if t is None else str(t) for t in textos]
    if not textos:
        raise ValueError("Lista de textos vazia: nada para calcular.")

    npy, meta = _caminhos(caminho_cache)
    digital = _impressao_digital(textos, modelo)

    if not forcar and npy.exists() and meta.exists():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if info.get("impressao_digital") == digital and info.get("n_textos") == len(textos):
                if verbose:
                    print(f"Embeddings carregados do cache: {npy}")
                return np.load(npy)
        except (json.JSONDecodeError, OSError):
            pass  # cache ilegivel: recalcula
        if verbose:
            print("Cache de embeddings desatualizado (textos ou modelo mudaram): recalculando.")

    if modelo == MODELO_TESTE_TFIDF:
        print("ATENCAO: usando vetores TF-IDF de TESTE. Nao use nos resultados.")
        emb = _embeddings_tfidf(textos)
    else:
        emb = _embeddings_modelo(textos, modelo, batch_size, verbose)

    if len(emb) != len(textos):
        raise RuntimeError("O numero de embeddings difere do numero de textos.")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, emb)
    meta.write_text(json.dumps({
        "modelo": modelo,
        "n_textos": len(textos),
        "dimensoes": int(emb.shape[1]),
        "impressao_digital": digital,
    }, indent=2), encoding="utf-8")
    if verbose:
        print(f"Embeddings salvos: {npy}  formato {emb.shape}")
    return emb


def main() -> None:
    ap = argparse.ArgumentParser(description="Calcula e salva os embeddings dos anuncios.")
    ap.add_argument("--tfidf", action="store_true", help="vetores de TESTE (sem baixar modelo)")
    ap.add_argument("--dedup", action="store_true", help="so um anuncio por grupo de copias")
    ap.add_argument("--forcar", action="store_true", help="ignora o cache")
    ap.add_argument("--modelo", default=None, help="nome de outro modelo do Hugging Face")
    args = ap.parse_args()

    from src.dados import obter_dados_limpos
    df = obter_dados_limpos()
    if args.dedup:
        from src.duplicatas import deduplicar
        df, _, _ = deduplicar(df)

    modelo = MODELO_TESTE_TFIDF if args.tfidf else (args.modelo or config.MODELO_EMBEDDING)
    caminho = config.DIR_SAIDA / ("embeddings_tfidf_teste.npy" if args.tfidf else "embeddings.npy")
    emb = gerar_embeddings(df["texto_limpo"].tolist(), modelo=modelo,
                           caminho_cache=caminho, forcar=args.forcar)
    print(f"Pronto: {emb.shape[0]} anuncios x {emb.shape[1]} dimensoes.")


if __name__ == "__main__":
    executar(main)
