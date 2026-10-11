"""
Deduplicacao de anuncios: exata + quase-duplicatas (simhash).

Por que isso importa: golpistas replicam o mesmo anuncio dezenas de vezes com
pequenas variacoes (preco, uma palavra, emoji). Se todas as copias entrarem no
BERTopic, elas viram "clusters" artificiais e distorcem a prevalencia.

Como funciona:
 1. Duplicata exata: mesmo texto depois de normalizar (minusculas, sem acento
    nem pontuacao). Emoji, HTML, link e "Patrocinado" ja foram tirados na limpeza.
 2. Quase-duplicata: simhash de 64 bits (Manku et al., 2007, tambem usado por
    Edelson et al., 2020). Dois textos sao candidatos se a distancia de Hamming
    entre os simhashes e <= DISTANCIA_SIMHASH. Para nao comparar todos contra
    todos, usa-se LSH por blocos de bits (principio da casa dos pombos: se a
    distancia e <= k, pelo menos um dos k+1 blocos e identico). Cada par
    candidato ainda precisa passar na similaridade de Jaccard das palavras
    (JACCARD_MINIMO), o que reduz falsos positivos.
 3. Os pares ligados formam grupos (componentes conexos, via union-find).
    Cada grupo tem um representante (o anuncio mais antigo).

Saidas importantes (colunas):
    grupo_dup, n_copias_grupo, n_variantes_grupo, n_anunciantes_grupo,
    eh_representante

Uso pelo terminal (na raiz do projeto):
    python -m src.duplicatas
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config
from src.cli import executar  # noqa: E402
from src.dados import normalizar_chave, obter_dados_limpos

BITS = 64
_popcount = int.bit_count if hasattr(int, "bit_count") else (lambda x: bin(x).count("1"))


# ---------------------------------------------------------------------------
# Simhash
# ---------------------------------------------------------------------------
def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def _tokens(chave: str) -> list[str]:
    return chave.split()


def simhash(tokens: list[str]) -> int:
    """Fingerprint de 64 bits: textos parecidos geram fingerprints parecidos."""
    if not tokens:
        return 0
    acumulador = np.zeros(BITS, dtype=np.int64)
    posicoes = np.arange(BITS, dtype=np.uint64)
    for token in tokens:
        h = np.uint64(_hash64(token))
        bits = ((h >> posicoes) & np.uint64(1)).astype(np.int64)
        acumulador += 2 * bits - 1          # bit 1 vota +1, bit 0 vota -1
    resultado = 0
    for i in range(BITS):
        if acumulador[i] > 0:
            resultado |= 1 << i
    return resultado


def distancia_hamming(a: int, b: int) -> int:
    return _popcount(a ^ b)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class _UniaoBusca:
    """Union-find simples para juntar pares parecidos em grupos."""

    def __init__(self, n: int):
        self.pai = list(range(n))

    def achar(self, i: int) -> int:
        while self.pai[i] != i:
            self.pai[i] = self.pai[self.pai[i]]
            i = self.pai[i]
        return i

    def unir(self, i: int, j: int) -> None:
        ri, rj = self.achar(i), self.achar(j)
        if ri != rj:
            self.pai[max(ri, rj)] = min(ri, rj)


def _pares_candidatos(fps: list[int], k: int):
    """Gera pares (i, j) cujos simhashes compartilham ao menos um bloco de bits."""
    blocos = np.array_split(np.arange(BITS), k + 1)
    mascaras = []
    for b in blocos:
        mascara, deslocamento = 0, int(b[0])
        for bit in b:
            mascara |= 1 << int(bit)
        mascaras.append((mascara, deslocamento))

    vistos: set[tuple[int, int]] = set()
    for idx_bloco, (mascara, desloc) in enumerate(mascaras):
        baldes: dict[int, list[int]] = defaultdict(list)
        for i, fp in enumerate(fps):
            if fp == 0:                      # texto sem palavras: ignora
                continue
            baldes[(fp & mascara) >> desloc].append(i)
        for membros in baldes.values():
            if len(membros) < 2:
                continue
            for a in range(len(membros)):
                for b in range(a + 1, len(membros)):
                    par = (membros[a], membros[b])
                    if par not in vistos:
                        vistos.add(par)
                        yield par


# ---------------------------------------------------------------------------
# Agrupamento
# ---------------------------------------------------------------------------
def agrupar_duplicatas(df: pd.DataFrame,
                       distancia_max: int = config.DISTANCIA_SIMHASH,
                       jaccard_minimo: float = config.JACCARD_MINIMO) -> pd.DataFrame:
    """
    Marca grupos de duplicatas exatas e quase-duplicatas. NAO remove linhas.
    Precisa da coluna 'texto_limpo' (ou 'texto_chave').
    """
    df = df.reset_index(drop=True).copy()
    if "texto_chave" not in df.columns:
        df["texto_chave"] = df["texto_limpo"].map(normalizar_chave)

    # 1) trabalhar so com textos unicos (duplicatas exatas ja ficam juntas)
    chaves = df["texto_chave"].drop_duplicates().tolist()
    conjuntos = [set(_tokens(c)) for c in chaves]
    fps = [simhash(_tokens(c)) for c in chaves]

    # 2) pares candidatos (LSH) confirmados por Hamming e Jaccard
    uf = _UniaoBusca(len(chaves))
    for i, j in _pares_candidatos(fps, distancia_max):
        if (distancia_hamming(fps[i], fps[j]) <= distancia_max
                and _jaccard(conjuntos[i], conjuntos[j]) >= jaccard_minimo):
            uf.unir(i, j)

    # 3) numerar grupos na ordem de aparicao
    raiz_para_grupo: dict[int, int] = {}
    chave_para_grupo: dict[str, int] = {}
    for i, chave in enumerate(chaves):
        raiz = uf.achar(i)
        if raiz not in raiz_para_grupo:
            raiz_para_grupo[raiz] = len(raiz_para_grupo) + 1
        chave_para_grupo[chave] = raiz_para_grupo[raiz]
    df["grupo_dup"] = df["texto_chave"].map(chave_para_grupo).astype(int)

    # 4) metricas por grupo
    g = df.groupby("grupo_dup")
    df["n_copias_grupo"] = g["grupo_dup"].transform("size").astype(int)
    df["n_variantes_grupo"] = g["texto_chave"].transform("nunique").astype(int)
    if "anunciante" in df.columns:
        df["n_anunciantes_grupo"] = g["anunciante"].transform(
            lambda s: s.nunique(dropna=True)).astype(int)

    # 5) representante = anuncio mais antigo do grupo (sem data: o primeiro da base)
    if "data" in df.columns:
        ordem = df.assign(_pos=np.arange(len(df))).sort_values(
            ["data", "_pos"], na_position="last", kind="stable").index
    else:
        ordem = df.index
    primeiro = pd.Series(~df.loc[ordem, "grupo_dup"].duplicated(keep="first").values,
                         index=ordem)
    df["eh_representante"] = primeiro.reindex(df.index)
    return df


def deduplicar(df: pd.DataFrame,
               distancia_max: int = config.DISTANCIA_SIMHASH,
               jaccard_minimo: float = config.JACCARD_MINIMO,
               verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Devolve (df_dedup, df_marcado, resumo).
      df_dedup    : um anuncio por grupo (o representante), com n_copias_grupo
      df_marcado  : todos os anuncios, com as colunas de grupo
      resumo      : numeros para o relatorio
    """
    marcado = agrupar_duplicatas(df, distancia_max, jaccard_minimo)
    dedup = marcado[marcado["eh_representante"]].reset_index(drop=True)
    n_total = len(marcado)
    n_unicos = marcado["texto_chave"].nunique()
    resumo = {
        "n_total": n_total,
        "n_removidos_exatos": n_total - n_unicos,
        "n_removidos_quase": n_unicos - len(dedup),
        "n_final": len(dedup),
        "pct_removido": round(100 * (n_total - len(dedup)) / n_total, 1) if n_total else 0.0,
        "grupos_com_varios_anunciantes": int(
            (dedup["n_anunciantes_grupo"] >= 2).sum()) if "n_anunciantes_grupo" in dedup else 0,
        "distancia_max": distancia_max,
        "jaccard_minimo": jaccard_minimo,
    }
    if verbose:
        print(f"Deduplicacao: {n_total} -> {len(dedup)} anuncios "
              f"({resumo['n_removidos_exatos']} duplicatas exatas + "
              f"{resumo['n_removidos_quase']} quase-duplicatas = {resumo['pct_removido']}% removido)")
        print(f"Grupos usados por mais de um anunciante: {resumo['grupos_com_varios_anunciantes']}")
    return dedup, marcado, resumo


def resumo_grupos(marcado: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Maiores grupos de copias (para olhar na mao e citar no texto)."""
    rep = marcado[marcado["eh_representante"]].set_index("grupo_dup")
    cols = {"n_copias_grupo": "n_anuncios", "n_variantes_grupo": "n_variantes"}
    saida = rep[list(cols)].rename(columns=cols)
    if "n_anunciantes_grupo" in rep:
        saida["n_anunciantes"] = rep["n_anunciantes_grupo"]
        saida["varios_anunciantes"] = saida["n_anunciantes"] >= 2
    saida["exemplo"] = rep["texto_limpo"].str.slice(0, 110)
    saida = saida[saida["n_anuncios"] >= 2].sort_values("n_anuncios", ascending=False)
    return saida.head(top).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicacao de anuncios (exata + simhash).")
    ap.add_argument("--distancia", type=int, default=config.DISTANCIA_SIMHASH)
    ap.add_argument("--jaccard", type=float, default=config.JACCARD_MINIMO)
    args = ap.parse_args()

    df = obter_dados_limpos()
    dedup, marcado, resumo = deduplicar(df, args.distancia, args.jaccard)
    print("\nMaiores grupos de copias:")
    print(resumo_grupos(marcado).to_string(index=False))

    config.garantir_pastas()
    marcado.to_csv(config.DIR_SAIDA / "anuncios_com_grupos.csv", index=False, encoding="utf-8")
    dedup.to_csv(config.DIR_SAIDA / "anuncios_dedup.csv", index=False, encoding="utf-8")
    print(f"\nSalvo em {config.DIR_SAIDA}")


if __name__ == "__main__":
    executar(main)
