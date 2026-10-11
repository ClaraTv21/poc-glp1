"""
Estatistica descritiva da base de anuncios: tabelas (CSV) e graficos (PNG).

Gera, em config.DIR_SAIDA:
    desc_estatisticas_texto.csv     tamanho dos textos, emojis
    desc_palavras_frequentes.csv    palavras mais comuns
    desc_bigramas.csv               pares de palavras mais comuns
    desc_por_mes.csv                anuncios por mes
    desc_anunciantes.csv            anuncios por anunciante (com codigo anonimo)
    desc_resumo.csv                 numeros-chave para o texto do relatorio
    fig_tamanho_textos.png, fig_palavras_frequentes.png, fig_anuncios_por_mes.png,
    fig_anunciantes.png, nuvem_palavras.png (precisa do pacote wordcloud)

Privacidade: nos GRAFICOS os anunciantes aparecem como "Anunciante 01, 02..."
(por ordem de volume). A tabela desc_anunciantes.csv guarda o nome real para
uso interno; no relatorio use apenas o codigo.

Uso pelo terminal (na raiz do projeto):
    python -m src.descritiva            # todos os anuncios
    python -m src.descritiva --dedup    # so um anuncio por grupo de copias
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.cli import executar  # noqa: E402
from src.dados import obter_dados_limpos  # noqa: E402

DPI = 200
COR = "#2F5D8C"


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def estatisticas_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo (media, mediana, quartis...) de caracteres, palavras e emojis."""
    cols = [c for c in ("n_chars", "n_palavras", "n_emojis") if c in df.columns]
    resumo = df[cols].describe().T
    resumo.index.name = "variavel"
    return resumo.round(2).reset_index()


def _ngramas(palavras: list[str], ngrama: int) -> list[str]:
    if ngrama == 1:
        return palavras
    return [" ".join(par) for par in zip(*(palavras[i:] for i in range(ngrama)))]


def _contar(df: pd.DataFrame, col: str, ngrama: int) -> Counter:
    """Frequencia total de cada termo (palavra ou par de palavras)."""
    contador: Counter = Counter()
    for texto in df[col].fillna("").astype(str):
        contador.update(_ngramas(texto.split(), ngrama))
    return contador


def palavras_frequentes(df: pd.DataFrame, n: int = 30, col: str = "texto_vetor",
                        ngrama: int = 1) -> pd.DataFrame:
    """
    Termos mais frequentes (ngrama=1: palavras; 2: pares de palavras).
    'n_anuncios' = em quantos anuncios o termo aparece (mais util que a contagem
    bruta, porque um anuncio que repete a palavra nao conta duas vezes).
    """
    frequencia: Counter = Counter()
    doc_freq: Counter = Counter()
    for texto in df[col].fillna("").astype(str):
        termos = _ngramas(texto.split(), ngrama)
        frequencia.update(termos)
        doc_freq.update(set(termos))
    total_docs = max(len(df), 1)
    linhas = [(t, f, doc_freq[t], round(100 * doc_freq[t] / total_docs, 1))
              for t, f in frequencia.most_common(n)]
    return pd.DataFrame(linhas, columns=["termo", "frequencia", "n_anuncios", "pct_anuncios"])


def anuncios_por_mes(df: pd.DataFrame) -> pd.DataFrame:
    """Quantidade de anuncios por mes (ignora datas invalidas)."""
    if "mes" not in df.columns or (df["mes"] == "").all():
        return pd.DataFrame(columns=["mes", "n_anuncios"])
    tabela = (df[df["mes"] != ""].groupby("mes").size()
              .rename("n_anuncios").reset_index().sort_values("mes"))
    return tabela


def anuncios_por_anunciante(df: pd.DataFrame) -> pd.DataFrame:
    """Anuncios por anunciante, com percentual acumulado e codigo anonimo."""
    if "anunciante" not in df.columns or df["anunciante"].isna().all():
        return pd.DataFrame(columns=["codigo", "anunciante", "n_anuncios", "pct", "pct_acumulado"])
    tabela = (df.groupby("anunciante").size().rename("n_anuncios")
              .sort_values(ascending=False, kind="stable").reset_index())
    tabela["pct"] = (100 * tabela["n_anuncios"] / tabela["n_anuncios"].sum()).round(2)
    tabela["pct_acumulado"] = tabela["pct"].cumsum().round(2)
    tabela.insert(0, "codigo", [f"Anunciante {i:02d}" for i in range(1, len(tabela) + 1)])
    return tabela


def numeros_chave(df: pd.DataFrame) -> pd.DataFrame:
    """Numeros prontos para citar no texto (formato: indicador, valor)."""
    linhas: list[tuple[str, object]] = [("n_anuncios", len(df))]
    if "texto_chave" in df.columns:
        linhas.append(("n_textos_unicos", int(df["texto_chave"].nunique())))
    if "n_palavras" in df.columns and len(df):
        linhas += [("palavras_media", round(float(df["n_palavras"].mean()), 1)),
                   ("palavras_mediana", float(df["n_palavras"].median()))]
    if "n_emojis" in df.columns and len(df):
        linhas.append(("pct_com_emoji", round(100 * float((df["n_emojis"] > 0).mean()), 1)))
    anunc = anuncios_por_anunciante(df)
    if len(anunc):
        linhas += [("n_anunciantes", len(anunc)),
                   ("anuncios_por_anunciante_media", round(float(anunc["n_anuncios"].mean()), 2)),
                   ("anuncios_por_anunciante_mediana", float(anunc["n_anuncios"].median())),
                   ("pct_dos_10_maiores_anunciantes", round(float(anunc["pct"].head(10).sum()), 1))]
    datas = df["data"].dropna() if "data" in df.columns else pd.Series(dtype="datetime64[ns]")
    if len(datas):
        linhas += [("data_mais_antiga", str(datas.min().date())),
                   ("data_mais_recente", str(datas.max().date())),
                   ("anuncios_sem_data_valida", int(df["data"].isna().sum()))]
    return pd.DataFrame(linhas, columns=["indicador", "valor"])


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def _salvar(fig, caminho: Path) -> Path:
    fig.tight_layout()
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return caminho


def grafico_tamanho_textos(df: pd.DataFrame, pasta: Path) -> Path | None:
    if "n_palavras" not in df.columns or df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["n_palavras"], bins=30, color=COR, edgecolor="white")
    ax.axvline(df["n_palavras"].median(), color="#B03A2E", linestyle="--",
               label=f"mediana = {df['n_palavras'].median():.0f}")
    ax.set_xlabel("Número de palavras do anúncio")
    ax.set_ylabel("Número de anúncios")
    ax.set_title("Distribuição do tamanho dos anúncios")
    ax.legend()
    return _salvar(fig, pasta / "fig_tamanho_textos.png")


def grafico_palavras_frequentes(tabela: pd.DataFrame, pasta: Path, n: int = 20) -> Path | None:
    if tabela.empty:
        return None
    t = tabela.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.5))
    ax.barh(t["termo"], t["pct_anuncios"], color=COR)
    ax.set_xlabel("% dos anúncios em que a palavra aparece")
    ax.set_title(f"As {len(t)} palavras mais frequentes")
    return _salvar(fig, pasta / "fig_palavras_frequentes.png")


def grafico_anuncios_por_mes(tabela: pd.DataFrame, pasta: Path) -> Path | None:
    if tabela.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tabela["mes"], tabela["n_anuncios"], marker="o", color=COR)
    ax.set_xlabel("Mês")
    ax.set_ylabel("Número de anúncios")
    ax.set_title("Anúncios por mês")
    passo = max(1, len(tabela) // 12)
    ax.set_xticks(range(0, len(tabela), passo))
    ax.set_xticklabels(tabela["mes"].iloc[::passo], rotation=45, ha="right")
    return _salvar(fig, pasta / "fig_anuncios_por_mes.png")


def grafico_anunciantes(tabela: pd.DataFrame, pasta: Path, n: int = 15) -> Path | None:
    if tabela.empty:
        return None
    t = tabela.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.5))
    ax.barh(t["codigo"], t["n_anuncios"], color=COR)   # codigo anonimo, nunca o nome real
    ax.set_xlabel("Número de anúncios")
    ax.set_title(f"Os {len(t)} anunciantes com mais anúncios")
    return _salvar(fig, pasta / "fig_anunciantes.png")


def nuvem_de_palavras(df: pd.DataFrame, pasta: Path, col: str = "texto_vetor",
                      nome: str = "nuvem_palavras.png") -> Path | None:
    """Nuvem de palavras (precisa de: pip install wordcloud). Pula se faltar."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("Aviso: pacote 'wordcloud' nao instalado; nuvem de palavras pulada.")
        return None
    contador = _contar(df, col, 1)
    if not contador:
        return None
    nuvem = WordCloud(width=1600, height=800, background_color="white",
                      collocations=False, random_state=config.SEED)
    nuvem.generate_from_frequencies(dict(contador))
    caminho = pasta / nome
    nuvem.to_file(str(caminho))
    return caminho


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------
def rodar_descritiva(df: pd.DataFrame, pasta: str | Path | None = None,
                     verbose: bool = True) -> dict:
    """Gera todas as tabelas e graficos. Devolve os caminhos e as tabelas."""
    pasta = Path(pasta or config.DIR_SAIDA)
    pasta.mkdir(parents=True, exist_ok=True)

    tabelas = {
        "desc_estatisticas_texto": estatisticas_texto(df),
        "desc_palavras_frequentes": palavras_frequentes(df, n=30, ngrama=1),
        "desc_bigramas": palavras_frequentes(df, n=30, ngrama=2),
        "desc_por_mes": anuncios_por_mes(df),
        "desc_anunciantes": anuncios_por_anunciante(df),
        "desc_resumo": numeros_chave(df),
    }
    for nome, tabela in tabelas.items():
        tabela.to_csv(pasta / f"{nome}.csv", index=False, encoding="utf-8-sig")

    figuras = {
        "tamanho_textos": grafico_tamanho_textos(df, pasta),
        "palavras_frequentes": grafico_palavras_frequentes(tabelas["desc_palavras_frequentes"], pasta),
        "anuncios_por_mes": grafico_anuncios_por_mes(tabelas["desc_por_mes"], pasta),
        "anunciantes": grafico_anunciantes(tabelas["desc_anunciantes"], pasta),
        "nuvem": nuvem_de_palavras(df, pasta),
    }
    if verbose:
        print("\nNUMEROS-CHAVE")
        print(tabelas["desc_resumo"].to_string(index=False))
        print("\nPALAVRAS MAIS FREQUENTES")
        print(tabelas["desc_palavras_frequentes"].head(10).to_string(index=False))
        gerados = [str(p.name) for p in figuras.values() if p is not None]
        print(f"\nArquivos gerados em {pasta}:\n  " + "\n  ".join(
            [f"{n}.csv" for n in tabelas] + gerados))
    return {"pasta": pasta, "tabelas": tabelas, "figuras": figuras}


def main() -> None:
    ap = argparse.ArgumentParser(description="Estatistica descritiva dos anuncios.")
    ap.add_argument("--dedup", action="store_true",
                    help="usa so um anuncio por grupo de copias")
    args = ap.parse_args()

    df = obter_dados_limpos()
    if args.dedup:
        from src.duplicatas import deduplicar
        df, _, _ = deduplicar(df)
    rodar_descritiva(df, config.DIR_SAIDA / ("descritiva_dedup" if args.dedup else "descritiva"))


if __name__ == "__main__":
    executar(main)
