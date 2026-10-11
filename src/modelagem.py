"""
Modelagem de topicos com BERTopic (embeddings -> UMAP -> HDBSCAN -> c-TF-IDF).

Pecas principais:
    parametros_padrao(n_docs)   sugere parametros conforme o tamanho da base
    carregar_insumos(...)       carrega anuncios + embeddings alinhados (usado por todos)
    construir_modelo(...)       monta o BERTopic com UMAP + HDBSCAN + vetorizador
    treinar(docs, embeddings)   ajusta o modelo e devolve (modelo, topicos)
    reduzir_outliers(...)       (opcional) realoca anuncios sem grupo (-1)
    estatisticas_modelo(...)    numero de topicos, % de outliers etc.
    resumo_topicos(...)         tabela com palavras-chave e exemplos de cada topico
    salvar_resultados(...)      grava CSVs e o modelo

Os 4 "botoes" que voce vai ajustar na etapa de hiperparametros:
    min_cluster_size (HDBSCAN)  tamanho minimo de um grupo
    min_samples (HDBSCAN)       quao rigoroso e para aceitar um grupo
    n_neighbors (UMAP)          quao "local" e a visao do modelo
    n_components (UMAP)         em quantas dimensoes comprimir

Topico -1 = "outliers": anuncios que nao entraram em nenhum grupo.

Uso pelo terminal (na raiz do projeto):
    python -m src.modelagem --tfidf --dedup     # teste rapido (vetores de TESTE)
    python -m src.modelagem --dedup             # modelo real (baixa o modelo na 1a vez)
    python -m src.modelagem --dedup --min-cluster-size 15 --reduzir-outliers
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.cli import executar  # noqa: E402
from src.limpeza import STOPWORDS_PT

# O sklearn avisa que algumas stopwords nao batem com seu tokenizador; e inofensivo.
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent")


# ---------------------------------------------------------------------------
# Montagem do modelo
# ---------------------------------------------------------------------------
def parametros_padrao(n_docs: int) -> dict:
    """
    Ponto de partida (nao e o valor final!). min_cluster_size ~ 1% da base,
    limitado entre 5 e 40: 500 anuncios -> 5; 4400 anuncios -> 40.
    """
    return {
        "min_cluster_size": int(np.clip(round(0.01 * n_docs), 5, 40)),
        "min_samples": None,          # None = HDBSCAN usa o mesmo valor de min_cluster_size
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "ngrama": (1, 2),
        "min_df": 2,
        "top_n_words": 10,
        "nr_topics": None,            # ex.: 12 para fundir topicos parecidos
    }


def criar_umap(n_docs: int, n_neighbors: int = 15, n_components: int = 5,
               min_dist: float = 0.0, seed: int = config.SEED):
    """
    Cria o UMAP. O grid.py e o modelo final usam ESTA funcao, entao os dois
    produzem exatamente os mesmos vetores reduzidos (mesma semente).
    n_neighbors e n_components sao limitados quando a base e muito pequena.
    """
    from umap import UMAP
    n_neighbors = int(max(2, min(n_neighbors, n_docs - 1)))
    n_components = int(max(2, min(n_components, n_docs - 2)))
    return UMAP(n_neighbors=n_neighbors, n_components=n_components,
                min_dist=min_dist, metric="cosine", random_state=seed)


def construir_modelo(n_docs: int, seed: int = config.SEED, umap_model=None, **ajustes):
    """
    Monta o BERTopic. Qualquer chave de parametros_padrao() pode ser sobrescrita:
        construir_modelo(450, min_cluster_size=8, n_neighbors=10)

    umap_model: normalmente None (cria o UMAP aqui). O grid.py passa
    BaseDimensionalityReduction() quando ja tem os vetores reduzidos prontos.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    p = {**parametros_padrao(n_docs), **{k: v for k, v in ajustes.items() if v is not None
                                         or k in ("min_samples", "nr_topics")}}
    invalidos = set(ajustes) - set(parametros_padrao(n_docs))
    if invalidos:
        raise ValueError(f"Parametros desconhecidos: {sorted(invalidos)}. "
                         f"Validos: {sorted(parametros_padrao(n_docs))}")

    if umap_model is None:
        umap_model = criar_umap(n_docs, p["n_neighbors"], p["n_components"], p["min_dist"], seed)
    hdbscan_model = HDBSCAN(min_cluster_size=int(p["min_cluster_size"]),
                            min_samples=p["min_samples"], metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    vetorizador = CountVectorizer(stop_words=sorted(STOPWORDS_PT), ngram_range=p["ngrama"],
                                  min_df=p["min_df"])
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vetorizador, top_n_words=p["top_n_words"],
                    nr_topics=p["nr_topics"], calculate_probabilities=False,
                    language="multilingual", verbose=False)


def treinar(docs: list[str], embeddings: np.ndarray, seed: int = config.SEED,
            umap_model=None, **ajustes):
    """
    Ajusta o BERTopic usando embeddings ja calculados.
    `docs` deve ser o texto limpo (frases completas) e estar na MESMA ordem
    das linhas de `embeddings`. Devolve (modelo, lista_de_topicos).
    """
    docs = ["" if d is None else str(d) for d in docs]
    embeddings = np.asarray(embeddings)
    if len(docs) != len(embeddings):
        raise ValueError(f"{len(docs)} textos mas {len(embeddings)} embeddings: "
                         "eles precisam estar alinhados (mesma ordem e tamanho).")
    modelo = construir_modelo(len(docs), seed=seed, umap_model=umap_model, **ajustes)
    topicos, _ = modelo.fit_transform(docs, embeddings=embeddings)
    return modelo, list(topicos)


def reduzir_outliers(modelo, docs: list[str], topicos: list[int], embeddings: np.ndarray,
                     estrategia: str = "embeddings", limiar: float = 0.0) -> list[int]:
    """
    Realoca anuncios do topico -1 para o topico mais proximo.
    estrategia: 'embeddings' (recomendada) ou 'c-tf-idf'.
    limiar: 0 realoca todos; valores maiores (0.3, 0.5...) so os bem parecidos.
    Atualiza as palavras-chave do modelo e devolve a nova lista de topicos.
    """
    if -1 not in topicos:
        return list(topicos)
    novos = modelo.reduce_outliers(
        docs, topicos, strategy=estrategia, threshold=limiar,
        embeddings=embeddings if estrategia == "embeddings" else None)
    modelo.update_topics(docs, topics=novos, vectorizer_model=modelo.vectorizer_model)
    return list(novos)


# ---------------------------------------------------------------------------
# Insumos e nomes de arquivos (compartilhados por grid, anotacao, figuras...)
# ---------------------------------------------------------------------------
def prefixo_resultados(usar_tfidf: bool = False) -> str:
    """Prefixo dos arquivos de resultado (separa teste TF-IDF do modelo real)."""
    return "bertopic_tfidf_teste" if usar_tfidf else "bertopic"


def caminho_embeddings(usar_tfidf: bool = False) -> Path:
    nome = "embeddings_tfidf_teste.npy" if usar_tfidf else "embeddings.npy"
    return config.DIR_SAIDA / nome


def carregar_insumos(usar_tfidf: bool = False, dedup: bool = True, forcar_embeddings: bool = False,
                     verbose: bool = True) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """
    Devolve (df, docs, embeddings), SEMPRE alinhados (mesma ordem e tamanho).
    dedup=True usa so um anuncio por grupo de copias (recomendado para modelar).
    """
    from src.dados import obter_dados_limpos
    from src.embeddings import MODELO_TESTE_TFIDF, gerar_embeddings

    df = obter_dados_limpos(verbose=False)
    if dedup:
        from src.duplicatas import deduplicar
        df, _, _ = deduplicar(df, verbose=verbose)
    df = df.reset_index(drop=True)
    docs = df["texto_limpo"].tolist()
    modelo_emb = MODELO_TESTE_TFIDF if usar_tfidf else config.MODELO_EMBEDDING
    emb = gerar_embeddings(docs, modelo=modelo_emb, caminho_cache=caminho_embeddings(usar_tfidf),
                           forcar=forcar_embeddings, verbose=verbose)
    return df, docs, emb


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
def estatisticas_modelo(topicos: list[int]) -> dict:
    """Numeros basicos para comparar modelos e para o relatorio."""
    t = pd.Series(topicos)
    validos = t[t != -1]
    tamanhos = validos.value_counts()
    return {
        "n_anuncios": int(len(t)),
        "n_topicos": int(tamanhos.size),
        "n_outliers": int((t == -1).sum()),
        "pct_outliers": round(100 * float((t == -1).mean()), 1) if len(t) else 0.0,
        "tamanho_mediano": float(tamanhos.median()) if tamanhos.size else 0.0,
        "maior_topico_pct": round(100 * float(tamanhos.max() / len(t)), 1) if tamanhos.size else 0.0,
    }


def resumo_topicos(modelo, topicos: list[int], df: pd.DataFrame | None = None,
                   n_palavras: int = 8, tam_exemplo: int = 140) -> pd.DataFrame:
    """
    Uma linha por topico: tamanho, palavras-chave e um anuncio representativo.
    Se `df` tiver 'n_copias_grupo' (vem de duplicatas.py), mostra tambem quantos
    anuncios o topico representa contando as copias.
    """
    t = pd.Series(topicos, name="topico")
    contagem = t.value_counts().sort_index()
    linhas = []
    for topico, n in contagem.items():
        palavras = [p for p, _ in (modelo.get_topic(int(topico)) or [])[:n_palavras]]
        reps = modelo.get_representative_docs(int(topico)) or []
        linha = {
            "topico": int(topico),
            "n_anuncios": int(n),
            "pct": round(100 * n / len(t), 1),
            "palavras_chave": ", ".join(palavras) if topico != -1 else "(outliers)",
            "exemplo": (reps[0][:tam_exemplo] if reps else ""),
        }
        if df is not None and "n_copias_grupo" in df.columns:
            linha["n_anuncios_com_copias"] = int(df.loc[t.values == topico, "n_copias_grupo"].sum())
        linhas.append(linha)
    return pd.DataFrame(linhas)


def salvar_resultados(modelo, df: pd.DataFrame, topicos: list[int],
                      pasta: str | Path | None = None, prefixo: str = "bertopic") -> dict:
    """Salva anuncios com topico, resumo dos topicos e o modelo (se der)."""
    pasta = Path(pasta or config.DIR_SAIDA)
    pasta.mkdir(parents=True, exist_ok=True)

    saida = df.copy()
    saida["topico"] = list(topicos)
    arq_anuncios = pasta / f"{prefixo}_anuncios_com_topicos.csv"
    saida.to_csv(arq_anuncios, index=False, encoding="utf-8")

    arq_resumo = pasta / f"{prefixo}_resumo_topicos.csv"
    resumo_topicos(modelo, topicos, df).to_csv(arq_resumo, index=False, encoding="utf-8-sig")

    arq_modelo = pasta / f"{prefixo}_modelo"
    try:
        modelo.save(str(arq_modelo), serialization="pickle", save_embedding_model=False)
    except Exception as erro:  # salvar o modelo e opcional: os CSVs ja bastam
        print(f"Aviso: nao consegui salvar o modelo ({erro}). Os CSVs foram salvos.")
        arq_modelo = None
    return {"anuncios": arq_anuncios, "resumo": arq_resumo, "modelo": arq_modelo}


# ---------------------------------------------------------------------------
# Execucao pelo terminal
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Treina o BERTopic nos anuncios.")
    ap.add_argument("--tfidf", action="store_true", help="vetores de TESTE (nao use nos resultados)")
    ap.add_argument("--dedup", action="store_true", help="modela so um anuncio por grupo de copias")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--n-neighbors", type=int, default=None)
    ap.add_argument("--n-components", type=int, default=None)
    ap.add_argument("--reduzir-outliers", action="store_true")
    args = ap.parse_args()

    modo = "DADOS INVENTADOS (mock)" if config.USAR_MOCK else "BASE REAL"
    print(f"Modo: {modo}")
    df, docs, emb = carregar_insumos(usar_tfidf=args.tfidf, dedup=args.dedup)

    ajustes = {"min_cluster_size": args.min_cluster_size, "min_samples": args.min_samples,
               "n_neighbors": args.n_neighbors, "n_components": args.n_components}
    modelo, topicos = treinar(docs, emb, **ajustes)
    print("\nANTES de reduzir outliers:", estatisticas_modelo(topicos))

    if args.reduzir_outliers:
        topicos = reduzir_outliers(modelo, docs, topicos, emb)
        print("DEPOIS de reduzir outliers:", estatisticas_modelo(topicos))

    resumo = resumo_topicos(modelo, topicos, df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\nTOPICOS ENCONTRADOS")
    print(resumo.to_string(index=False))

    if "narrativa_sintetica" in df.columns:   # so existe nos dados INVENTADOS
        print("\n[TESTE] Topicos x narrativa inventada (cada linha deveria ter 1 coluna dominante):")
        print(pd.crosstab(pd.Series(topicos, name="topico"), df["narrativa_sintetica"]).to_string())

    arquivos = salvar_resultados(modelo, df, topicos,
                                 prefixo=prefixo_resultados(args.tfidf))
    print("\nSalvo:", {k: str(v) for k, v in arquivos.items() if v})


if __name__ == "__main__":
    executar(main)
