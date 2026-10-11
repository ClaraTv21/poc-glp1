"""
Avaliacao de modelos de topicos: como saber se um modelo e "bom"?

Metricas (uma linha de tabela por modelo):
    n_topicos, pct_outliers, tamanho_mediano, maior_topico_pct
    coerencia      os termos de cada topico aparecem juntos nos mesmos anuncios?
                   (mais alto = topicos mais coerentes)
    diversidade    proporcao de termos DIFERENTES entre os topicos
                   (1.0 = nenhum termo se repete; perto de 0 = topicos redundantes)

Coerencia - tres medidas:
    "npmi"     (padrao) NPMI por anuncio, calculada aqui, sem dependencias e rapida
               (e a usada no grid). Vai de -1 a +1.
               Para cada par de termos de um topico: NPMI = log(p(a,b)/(p(a)p(b))) / -log(p(a,b)),
               com p = fracao dos anuncios que contem o termo. Pares em que um termo
               contem o outro ("sem" e "sem receita") sao ignorados, senao inflam a nota.
    "c_v"      medida classica de Roder et al. (2015), via gensim (mais lenta)
    "c_npmi"   NPMI com janela deslizante, via gensim
    As duas do gensim usam so palavras soltas (unigramas) e exigem: pip install gensim

Atencao: as metricas ajudam a COMPARAR modelos, nao substituem ler os topicos.

Tambem tem:
    estabilidade()    treina com varias sementes e mede se os grupos se repetem (ARI/AMI)
    escolher_modelo() aplica uma regra de escolha ESCRITA ANTES de ver os resultados

Uso pelo terminal (na raiz do projeto):
    python -m src.avaliacao --tfidf --dedup                 # avalia o modelo padrao
    python -m src.avaliacao --dedup --medida c_v            # coerencia c_v (gensim)
    python -m src.avaliacao --dedup --estabilidade          # testa 3 sementes
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from src import config
from src.cli import executar  # noqa: E402
from src.modelagem import carregar_insumos, estatisticas_modelo, treinar

MEDIDAS = ("npmi", "c_v", "c_npmi")
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Termos de cada topico
# ---------------------------------------------------------------------------
def termos_dos_topicos(modelo, n_palavras: int = 10) -> dict[int, list[str]]:
    """Palavras-chave de cada topico (sem o -1), na ordem de importancia."""
    termos: dict[int, list[str]] = {}
    for topico in sorted(t for t in modelo.get_topics() if t != -1):
        palavras = [p for p, _ in (modelo.get_topic(topico) or [])[:n_palavras] if p]
        if palavras:
            termos[int(topico)] = palavras
    return termos


def conjuntos_por_anuncio(modelo, docs: list[str]) -> list[set[str]]:
    """
    Para cada anuncio, o conjunto de termos (palavras e pares de palavras) que ele
    contem, calculado com o MESMO vetorizador do modelo (mesmas stopwords e n-gramas).
    """
    analisador = modelo.vectorizer_model.build_analyzer()
    return [set(analisador(d)) for d in docs]


# ---------------------------------------------------------------------------
# Coerencia
# ---------------------------------------------------------------------------
def coerencia_npmi(termos_por_topico: dict[int, list[str]],
                   conjuntos_docs: list[set[str]]) -> tuple[float, dict[int, float]]:
    """
    NPMI medio por anuncio. Devolve (media dos topicos, {topico: coerencia}).
    Topicos com menos de 2 pares validos ficam de fora da media.
    """
    n_docs = len(conjuntos_docs)
    todos = sorted({t for ts in termos_por_topico.values() for t in ts})
    if n_docs == 0 or not todos:
        return float("nan"), {}
    indice = {t: i for i, t in enumerate(todos)}

    presenca = np.zeros((len(todos), n_docs), dtype=np.float32)   # termo x anuncio
    for j, conjunto in enumerate(conjuntos_docs):
        for t in conjunto & indice.keys():
            presenca[indice[t], j] = 1.0
    freq = presenca.sum(axis=1)                  # em quantos anuncios cada termo aparece
    co = presenca @ presenca.T                   # em quantos anuncios cada PAR aparece

    por_topico: dict[int, float] = {}
    for topico, termos in termos_por_topico.items():
        valores = []
        for a, b in itertools.combinations(termos, 2):
            ta, tb = set(a.split()), set(b.split())
            if ta <= tb or tb <= ta:             # um termo contem o outro: ignora
                continue
            ia, ib = indice[a], indice[b]
            if freq[ia] == 0 or freq[ib] == 0:
                continue
            p_ab = co[ia, ib] / n_docs
            if p_ab <= 0:
                valores.append(-1.0)             # nunca aparecem juntos
            elif p_ab >= 1 - _EPS:
                valores.append(1.0)              # aparecem sempre juntos
            else:
                p_a, p_b = freq[ia] / n_docs, freq[ib] / n_docs
                valores.append(float(np.log((p_ab + _EPS) / (p_a * p_b)) / -np.log(p_ab + _EPS)))
        if len(valores) >= 1:
            por_topico[topico] = float(np.mean(valores))
    media = float(np.mean(list(por_topico.values()))) if por_topico else float("nan")
    return media, por_topico


def coerencia_gensim(modelo, docs: list[str], medida: str = "c_v",
                     n_palavras: int = 10) -> float:
    """Coerencia c_v ou c_npmi pelo gensim (so com palavras soltas)."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError as erro:
        raise ImportError("Para usar c_v ou c_npmi instale: pip install gensim "
                          "(ou use --medida npmi)") from erro
    from sklearn.feature_extraction.text import CountVectorizer

    base = modelo.vectorizer_model
    analisador = CountVectorizer(stop_words=base.stop_words, ngram_range=(1, 1),
                                 min_df=1).build_analyzer()
    textos = [analisador(d) for d in docs]
    dicionario = Dictionary(textos)

    topicos_palavras = []
    for topico in sorted(t for t in modelo.get_topics() if t != -1):
        unigramas = [p for p, _ in (modelo.get_topic(topico) or []) if " " not in p]
        unigramas = [p for p in unigramas if p in dicionario.token2id][:n_palavras]
        if len(unigramas) >= 2:
            topicos_palavras.append(unigramas)
    if not topicos_palavras:
        return float("nan")
    cm = CoherenceModel(topics=topicos_palavras, texts=textos, dictionary=dicionario,
                        coherence=medida, processes=1)
    return float(cm.get_coherence())


def diversidade_topicos(termos_por_topico: dict[int, list[str]]) -> float:
    """Termos unicos / termos totais (Dieng et al., 2020). 1.0 = sem repeticao."""
    todos = [t for ts in termos_por_topico.values() for t in ts]
    return round(len(set(todos)) / len(todos), 4) if todos else float("nan")


# ---------------------------------------------------------------------------
# Avaliacao de um modelo
# ---------------------------------------------------------------------------
def avaliar_modelo(modelo, topicos: list[int], docs: list[str], n_palavras: int = 10,
                   medida: str = "npmi", conjuntos_docs: list[set[str]] | None = None) -> dict:
    """
    Uma linha de metricas. conjuntos_docs (opcional) evita recalcular os termos
    de cada anuncio quando voce avalia muitos modelos com o mesmo vetorizador.
    """
    if medida not in MEDIDAS:
        raise ValueError(f"medida deve ser uma de {MEDIDAS}")
    linha = estatisticas_modelo(topicos)
    termos = termos_dos_topicos(modelo, n_palavras)

    if medida == "npmi":
        if conjuntos_docs is None:
            conjuntos_docs = conjuntos_por_anuncio(modelo, docs)
        coerencia, _ = coerencia_npmi(termos, conjuntos_docs)
    else:
        coerencia = coerencia_gensim(modelo, docs, medida, n_palavras)

    linha["coerencia"] = round(coerencia, 4) if coerencia == coerencia else float("nan")
    linha["medida_coerencia"] = medida
    linha["diversidade"] = diversidade_topicos(termos)
    return linha


def coerencia_por_topico(modelo, docs: list[str], n_palavras: int = 10) -> pd.DataFrame:
    """Coerencia NPMI de cada topico (para achar topicos "bagunçados")."""
    termos = termos_dos_topicos(modelo, n_palavras)
    _, por_topico = coerencia_npmi(termos, conjuntos_por_anuncio(modelo, docs))
    return (pd.DataFrame({"topico": list(por_topico), "coerencia_npmi": list(por_topico.values())})
            .sort_values("coerencia_npmi").reset_index(drop=True).round(4))


# ---------------------------------------------------------------------------
# Estabilidade e escolha
# ---------------------------------------------------------------------------
def estabilidade(docs: list[str], embeddings: np.ndarray, sementes: tuple[int, ...] = (42, 43, 44),
                 **ajustes) -> dict:
    """
    Treina o mesmo modelo com sementes diferentes e compara os agrupamentos
    dois a dois. ARI/AMI = 1 -> agrupamentos identicos; perto de 0 -> instavel.
    (Valores acima de ~0.7 costumam ser considerados estaveis.)
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    rotulos, n_topicos = [], []
    for semente in sementes:
        _, t = treinar(docs, embeddings, seed=semente, **ajustes)
        rotulos.append(np.array(t))
        n_topicos.append(int(len(set(t) - {-1})))
    aris, amis = [], []
    for a, b in itertools.combinations(range(len(sementes)), 2):
        aris.append(adjusted_rand_score(rotulos[a], rotulos[b]))
        amis.append(adjusted_mutual_info_score(rotulos[a], rotulos[b]))
    return {
        "sementes": list(sementes),
        "n_topicos_por_semente": n_topicos,
        "ari_medio": round(float(np.mean(aris)), 3),
        "ari_minimo": round(float(np.min(aris)), 3),
        "ami_medio": round(float(np.mean(amis)), 3),
    }


def escolher_modelo(tabela: pd.DataFrame, max_outliers: float = 30.0, min_topicos: int = 3,
                    max_topicos: int | None = None,
                    margem_empate: float = 0.01) -> tuple[pd.Series, str]:
    """
    Regra de escolha (registre em docs/decisoes.md ANTES de ver os resultados):
      1. fica so quem tem pct_outliers <= max_outliers e n_topicos entre min e max;
      2. entre esses, vence a MAIOR coerencia;
      3. se a diferenca de coerencia for menor que margem_empate, vence o de menos outliers;
      4. persistindo o empate, o de mais topicos (mais detalhe).
    Se ninguem passar no filtro, escolhe o de menos outliers e avisa.
    Devolve (linha escolhida, explicacao em texto).
    """
    t = tabela.dropna(subset=["coerencia"]).copy()
    ok = t[(t["pct_outliers"] <= max_outliers) & (t["n_topicos"] >= min_topicos)]
    if max_topicos is not None:
        ok = ok[ok["n_topicos"] <= max_topicos]

    if ok.empty:
        escolhido = t.sort_values(["pct_outliers", "coerencia"], ascending=[True, False]).iloc[0]
        return escolhido, (f"Nenhum modelo atendeu a regra (outliers <= {max_outliers}%, "
                           f">= {min_topicos} topicos). Escolhi o de MENOS outliers; "
                           "reveja a grade ou a regra.")
    melhor = ok["coerencia"].max()
    finalistas = ok[ok["coerencia"] >= melhor - margem_empate]
    escolhido = finalistas.sort_values(["pct_outliers", "n_topicos"],
                                       ascending=[True, False]).iloc[0]
    texto = (f"Regra: outliers <= {max_outliers}%, >= {min_topicos} topicos"
             + (f", <= {max_topicos} topicos" if max_topicos else "")
             + f"; maior coerencia (empate se diferenca < {margem_empate}). "
             f"{len(ok)} de {len(t)} modelos passaram no filtro; "
             f"{len(finalistas)} ficaram no empate.")
    return escolhido, texto


# ---------------------------------------------------------------------------
# Execucao pelo terminal
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Avalia o modelo de topicos padrao.")
    ap.add_argument("--tfidf", action="store_true", help="vetores de TESTE")
    ap.add_argument("--dedup", action="store_true", help="so um anuncio por grupo de copias")
    ap.add_argument("--medida", choices=MEDIDAS, default="npmi")
    ap.add_argument("--estabilidade", action="store_true", help="testa 3 sementes")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--n-neighbors", type=int, default=None)
    ap.add_argument("--n-components", type=int, default=None)
    args = ap.parse_args()

    df, docs, emb = carregar_insumos(usar_tfidf=args.tfidf, dedup=args.dedup)
    ajustes = {"min_cluster_size": args.min_cluster_size, "n_neighbors": args.n_neighbors,
               "n_components": args.n_components}
    modelo, topicos = treinar(docs, emb, **ajustes)

    metricas = avaliar_modelo(modelo, topicos, docs, medida=args.medida)
    print("\nMETRICAS DO MODELO")
    for chave, valor in metricas.items():
        print(f"  {chave:20s} {valor}")

    print("\nTOPICOS MENOS COERENTES (olhe estes primeiro):")
    print(coerencia_por_topico(modelo, docs).head(5).to_string(index=False))

    if args.estabilidade:
        print("\nESTABILIDADE (3 sementes)...")
        for chave, valor in estabilidade(docs, emb, **ajustes).items():
            print(f"  {chave:24s} {valor}")


if __name__ == "__main__":
    executar(main)
