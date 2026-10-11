"""
Grade de hiperparametros (UMAP + HDBSCAN) para o BERTopic.

Testa varias combinacoes, mede cada modelo (avaliacao.py) e grava tudo em
    outputs/<modo>/grid_resultados.csv
O grid e RETOMAVEL: se for interrompido (ou voce ampliar a grade), as
combinacoes ja feitas nao sao refeitas. Se a base ou os embeddings mudarem, o
arquivo antigo e guardado como *_anterior.csv e o grid recomeca.

Por que e rapido: o UMAP (a parte lenta) e calculado UMA vez para cada par
(n_neighbors, n_components) e reaproveitado por todas as combinacoes de
HDBSCAN. Isso da exatamente o mesmo resultado que treinar o modelo do zero
(testado: os topicos sao identicos).

Os 4 botoes:
    min_cluster_size   tamanho minimo de um grupo (menor = mais grupos, menores)
    min_samples        rigor para aceitar um grupo (menor = menos outliers)
    n_neighbors        UMAP: visao local (pequeno) ou global (grande)
    n_components       UMAP: em quantas dimensoes comprimir

Uso pelo terminal (na raiz do projeto):
    python -m src.grid --tfidf --dedup --rapido        # teste de 4 combinacoes
    python -m src.grid --dedup                         # grade completa (modelo real)
    python -m src.grid --dedup --max-outliers 25 --min-topicos 5
    python -m src.grid --dedup --grade '{"min_cluster_size":[8,12],"n_neighbors":[10,15]}'
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.cli import executar  # noqa: E402
from src.avaliacao import avaliar_modelo, conjuntos_por_anuncio, escolher_modelo
from src.modelagem import (carregar_insumos, construir_modelo, criar_umap,
                           parametros_padrao)

COLUNAS_PARAMETROS = ["min_cluster_size", "min_samples", "n_neighbors", "n_components"]


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------
def grade_padrao(n_docs: int, rapido: bool = False) -> dict:
    """
    Grade sugerida, proporcional ao tamanho da base (min_cluster_size parte de
    ~1% da base). Ex.: 450 anuncios -> [3, 5, 8, 10]; 4400 -> [20, 40, 60, 80].
    """
    base = parametros_padrao(n_docs)["min_cluster_size"]
    if rapido:
        return {"min_cluster_size": sorted({max(3, base), max(3, 2 * base)}),
                "min_samples_fracao": [1.0],
                "n_neighbors": [10, 15], "n_components": [5]}
    return {"min_cluster_size": sorted({max(3, round(base * f)) for f in (0.5, 1, 1.5, 2)}),
            "min_samples_fracao": [1.0, 0.5],
            "n_neighbors": [10, 15, 30], "n_components": [5, 10]}


def expandir_grade(grade: dict) -> list[dict]:
    """
    Transforma a grade em uma lista de combinacoes completas.
    Chaves aceitas: min_cluster_size, n_neighbors, n_components e
    min_samples (valores inteiros) OU min_samples_fracao (fracoes de min_cluster_size).
    Sem nenhum dos dois, min_samples = min_cluster_size.
    """
    desconhecidas = set(grade) - {"min_cluster_size", "min_samples", "min_samples_fracao",
                                  "n_neighbors", "n_components"}
    if desconhecidas:
        raise ValueError(f"Chaves desconhecidas na grade: {sorted(desconhecidas)}")
    combos, vistos = [], set()
    for mcs in grade.get("min_cluster_size", [5]):
        if "min_samples" in grade:
            amostras = {int(m) for m in grade["min_samples"]}
        else:
            fracoes = grade.get("min_samples_fracao", [1.0])
            amostras = {max(1, round(mcs * f)) for f in fracoes}
        for ms in sorted(amostras):
            for nn, nc in itertools.product(grade.get("n_neighbors", [15]),
                                            grade.get("n_components", [5])):
                chave = (int(mcs), int(ms), int(nn), int(nc))
                if chave not in vistos:
                    vistos.add(chave)
                    combos.append({"min_cluster_size": chave[0], "min_samples": chave[1],
                                   "n_neighbors": chave[2], "n_components": chave[3]})
    return combos


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def _base_id(docs: list[str], embeddings: np.ndarray) -> str:
    """Impressao digital da base: se textos ou embeddings mudarem, o grid recomeca."""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.encode("utf-8"))
        h.update(b"\x00")
    h.update(np.ascontiguousarray(embeddings).tobytes())
    return h.hexdigest()[:10]


def _chave(linha) -> tuple:
    return tuple(int(linha[c]) for c in COLUNAS_PARAMETROS)


def rodar_grid(docs: list[str], embeddings: np.ndarray, grade: dict | None = None,
               medida: str = "npmi", arquivo: str | Path | None = None,
               forcar: bool = False, seed: int = config.SEED,
               verbose: bool = True) -> pd.DataFrame:
    """Roda a grade, grava incrementalmente no CSV e devolve a tabela completa."""
    from bertopic.dimensionality import BaseDimensionalityReduction

    docs = ["" if d is None else str(d) for d in docs]
    embeddings = np.asarray(embeddings)
    if len(docs) != len(embeddings):
        raise ValueError(f"{len(docs)} textos mas {len(embeddings)} embeddings.")
    n_docs = len(docs)
    arquivo = Path(arquivo or config.DIR_SAIDA / "grid_resultados.csv")
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    combos = expandir_grade(grade or grade_padrao(n_docs))
    base = _base_id(docs, embeddings)

    # retomada: reaproveita linhas da MESMA base
    feitos: set[tuple] = set()
    if arquivo.exists() and not forcar:
        antigo = pd.read_csv(arquivo)
        if "base_id" in antigo.columns and (antigo["base_id"].astype(str) == base).all() \
                and (antigo["seed"] == seed).all():
            feitos = {_chave(l) for _, l in antigo.iterrows()}
        else:
            backup = arquivo.with_name(arquivo.stem + "_anterior.csv")
            arquivo.replace(backup)
            if verbose:
                print(f"A base, os embeddings ou a semente mudaram: grid antigo guardado em {backup.name}.")
    elif arquivo.exists() and forcar:
        arquivo.unlink()

    pendentes = [c for c in combos if tuple(c[k] for k in COLUNAS_PARAMETROS) not in feitos]
    if verbose:
        print(f"Grid: {len(combos)} combinacoes ({len(feitos)} ja feitas, {len(pendentes)} a fazer). "
              f"Medida de coerencia: {medida}.")

    conjuntos = None   # termos por anuncio: calculados 1 vez (o vetorizador e o mesmo em todo o grid)
    grupos_umap = sorted({(c["n_neighbors"], c["n_components"]) for c in pendentes})
    feitos_agora = 0
    for nn, nc in grupos_umap:
        t0 = time.time()
        umap = criar_umap(n_docs, nn, nc, seed=seed)
        umap.fit(embeddings)
        reduzido = umap.transform(embeddings)        # mesma sequencia que o BERTopic usa
        seg_umap = time.time() - t0

        do_grupo = [x for x in pendentes if (x["n_neighbors"], x["n_components"]) == (nn, nc)]
        for c in do_grupo:
            t1 = time.time()
            modelo = construir_modelo(n_docs, seed=seed, umap_model=BaseDimensionalityReduction(),
                                      min_cluster_size=c["min_cluster_size"],
                                      min_samples=c["min_samples"])
            topicos, _ = modelo.fit_transform(docs, embeddings=reduzido)
            if conjuntos is None and medida == "npmi":
                conjuntos = conjuntos_por_anuncio(modelo, docs)
            metricas = avaliar_modelo(modelo, list(topicos), docs, medida=medida,
                                      conjuntos_docs=conjuntos)
            linha = {**c, "seed": seed, "n_docs": n_docs, "base_id": base, **metricas,
                     "segundos": round(time.time() - t1 + seg_umap / len(do_grupo), 2)}
            pd.DataFrame([linha]).to_csv(arquivo, mode="a", index=False,
                                         header=not arquivo.exists(), encoding="utf-8")
            feitos_agora += 1
            if verbose:
                print(f"  [{feitos_agora}/{len(pendentes)}] mcs={c['min_cluster_size']:>3} "
                      f"ms={c['min_samples']:>3} nn={nn:>3} nc={nc:>2} -> "
                      f"{metricas['n_topicos']:>3} topicos, {metricas['pct_outliers']:>5.1f}% outliers, "
                      f"coerencia {metricas['coerencia']}")

    tabela = pd.read_csv(arquivo)
    tabela = tabela[tabela["base_id"].astype(str) == base]
    chaves_pedidas = {tuple(c[k] for k in COLUNAS_PARAMETROS) for c in combos}
    tabela = tabela[[(_chave(l) in chaves_pedidas) for _, l in tabela.iterrows()]]
    return tabela.sort_values("coerencia", ascending=False).reset_index(drop=True)


def parametros_do_modelo(linha: pd.Series) -> dict:
    """Extrai os 4 parametros de uma linha da tabela (para treinar o modelo final)."""
    return {c: int(linha[c]) for c in COLUNAS_PARAMETROS}


def comando_para_treinar(parametros: dict, dedup: bool = True, tfidf: bool = False) -> str:
    partes = ["python -m src.modelagem"]
    if dedup:
        partes.append("--dedup")
    if tfidf:
        partes.append("--tfidf")
    partes += [f"--min-cluster-size {parametros['min_cluster_size']}",
               f"--min-samples {parametros['min_samples']}",
               f"--n-neighbors {parametros['n_neighbors']}",
               f"--n-components {parametros['n_components']}"]
    return " ".join(partes)


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Grade de hiperparametros do BERTopic.")
    ap.add_argument("--tfidf", action="store_true", help="vetores de TESTE")
    ap.add_argument("--dedup", action="store_true", help="so um anuncio por grupo de copias")
    ap.add_argument("--rapido", action="store_true", help="grade minima (so para testar)")
    ap.add_argument("--medida", choices=("npmi", "c_v", "c_npmi"), default="npmi")
    ap.add_argument("--forcar", action="store_true", help="apaga o grid anterior e refaz")
    ap.add_argument("--grade", type=str, default=None,
                    help='grade propria em JSON, ex.: \'{"min_cluster_size":[8,12],"n_neighbors":[10,15]}\'')
    ap.add_argument("--max-outliers", type=float, default=30.0)
    ap.add_argument("--min-topicos", type=int, default=3)
    ap.add_argument("--max-topicos", type=int, default=None)
    args = ap.parse_args()

    modo = "DADOS INVENTADOS (mock)" if config.USAR_MOCK else "BASE REAL"
    print(f"Modo: {modo}")
    df, docs, emb = carregar_insumos(usar_tfidf=args.tfidf, dedup=args.dedup)

    grade = json.loads(args.grade) if args.grade else grade_padrao(len(docs), rapido=args.rapido)
    sufixo = ("" if args.medida == "npmi" else f"_{args.medida}") + ("_tfidf_teste" if args.tfidf else "")
    arquivo = config.DIR_SAIDA / f"grid_resultados{sufixo}.csv"
    tabela = rodar_grid(docs, emb, grade, medida=args.medida, arquivo=arquivo, forcar=args.forcar)

    colunas = ["min_cluster_size", "min_samples", "n_neighbors", "n_components", "n_topicos",
               "pct_outliers", "coerencia", "diversidade"]
    pd.set_option("display.width", 200)
    print("\nMELHORES 10 (por coerencia, sem filtro):")
    print(tabela[colunas].head(10).to_string(index=False))

    escolhido, regra = escolher_modelo(tabela, args.max_outliers, args.min_topicos, args.max_topicos)
    params = parametros_do_modelo(escolhido)
    print("\nMODELO ESCOLHIDO PELA REGRA")
    print(" ", regra)
    print(" ", {**params, "n_topicos": int(escolhido["n_topicos"]),
                "pct_outliers": float(escolhido["pct_outliers"]),
                "coerencia": float(escolhido["coerencia"])})

    arq_escolhido = config.DIR_SAIDA / f"grid_escolhido{sufixo}.json"
    arq_escolhido.write_text(json.dumps({
        "parametros": params, "regra": regra, "medida_coerencia": args.medida,
        "metricas": {k: (float(escolhido[k]) if k != "n_topicos" else int(escolhido[k]))
                     for k in ("n_topicos", "pct_outliers", "coerencia", "diversidade")},
        "dedup": args.dedup, "tfidf_teste": args.tfidf,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTabela completa: {arquivo}")
    print(f"Escolha salva em: {arq_escolhido}")
    print("\nProximo passo: LEIA os topicos do modelo escolhido antes de confiar na metrica:")
    print("  " + comando_para_treinar(params, dedup=args.dedup, tfidf=args.tfidf))


if __name__ == "__main__":
    executar(main)
