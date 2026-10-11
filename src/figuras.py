"""
Figuras finais do relatorio (PNG, 300 dpi) em outputs/<modo>/figuras/.

Cada figura usa o que ja existir e e PULADA (com aviso) se faltar um insumo:

  grid_hiperparametros.png      grid: coerencia x outliers; efeito de min_cluster_size
  tamanho_topicos.png           tamanho de cada topico e suas palavras-chave
  mapa_topicos.png              mapa 2D dos anuncios (UMAP) colorido por topico
  prevalencia_categorias.png    % de anuncios por categoria de risco (precisa anotacao)
  categorias_no_tempo.png       anuncios por mes e categoria          (precisa anotacao)
  taticas.png                   % de anuncios com cada tatica         (precisa anotacao)
  categoria_x_tatica.png        mapa de calor categoria x tatica      (precisa anotacao)
  nuvens/nuvem_<categoria>.png  nuvem de palavras de cada categoria   (precisa anotacao)

"precisa anotacao" = depois de preencher a planilha e rodar:
    python -m src.anotacao aplicar

SEGURANCA: quando config.USAR_MOCK e True, toda figura recebe a marca
"DADOS INVENTADOS - TESTE", para ninguem usar por engano no relatorio.
Os graficos nao mostram nomes de anunciantes.

Uso pelo terminal (na raiz do projeto):
    python -m src.figuras --tfidf          # figuras do teste (vetores de TESTE)
    python -m src.figuras                  # figuras do modelo real
    python -m src.figuras --sem-mapa       # pula o mapa 2D (mais rapido)
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.cli import executar  # noqa: E402
from src.anotacao import COL_TATICAS, EIXO2, SEM_ROTULO, caminho_rotulados  # noqa: E402
from src.modelagem import caminho_embeddings, prefixo_resultados  # noqa: E402

DPI = 300
COR = "#2F5D8C"
COR_OUTLIER = "#B0B0B0"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def pasta_figuras() -> Path:
    pasta = config.DIR_SAIDA / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


_MARCA: str | None = None   # texto do carimbo; definido por gerar_todas()


def _texto_marca(usar_tfidf: bool = False) -> str | None:
    """Carimbo de seguranca: dados inventados OU vetores de teste nunca vao ao relatorio."""
    if config.USAR_MOCK:
        return "DADOS INVENTADOS - TESTE"
    if usar_tfidf:
        return "VETORES DE TESTE - NÃO USAR"
    return None


def _salvar(fig, nome: str, pasta: Path | None = None) -> Path:
    """Salva a figura (com carimbo de seguranca quando for teste)."""
    marca = _MARCA if _MARCA is not None else _texto_marca()
    if marca:
        fig.text(0.5, 0.5, marca, rotation=30, fontsize=34, color="red",
                 alpha=0.16, ha="center", va="center", weight="bold", zorder=100)
    caminho = (pasta or pasta_figuras()) / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def _curto(palavras: str, n: int = 3) -> str:
    partes = [p.strip() for p in str(palavras).split(",") if p.strip()]
    return ", ".join(partes[:n])


def _cores(n: int):
    mapa = plt.get_cmap("tab20")
    return [mapa(i % 20) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Grid
# ---------------------------------------------------------------------------
def fig_grid(tabela: pd.DataFrame, escolhido: dict | None = None) -> Path | None:
    """(a) coerencia x % de outliers; (b) coerencia media por min_cluster_size."""
    if tabela.empty or tabela["coerencia"].isna().all():
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    pontos = ax1.scatter(tabela["pct_outliers"], tabela["coerencia"], c=tabela["n_topicos"],
                         cmap="viridis", s=45, edgecolor="white", linewidth=0.5)
    fig.colorbar(pontos, ax=ax1, label="Número de tópicos")
    if escolhido:
        p = escolhido["parametros"]
        sel = tabela[(tabela[list(p)] == pd.Series(p)).all(axis=1)]
        if len(sel):
            ax1.scatter(sel["pct_outliers"], sel["coerencia"], s=220, facecolor="none",
                        edgecolor="#B03A2E", linewidth=2, label="modelo escolhido")
            ax1.legend(loc="best")
    ax1.set_xlabel("% de anúncios sem tópico (outliers)")
    ax1.set_ylabel(f"Coerência ({tabela['medida_coerencia'].iloc[0]})")
    ax1.set_title("(a) Cada ponto é um modelo testado")

    for nn, sub in tabela.groupby("n_neighbors"):
        medias = sub.groupby("min_cluster_size")["coerencia"].mean()
        ax2.plot(medias.index, medias.values, marker="o", label=f"n_neighbors = {nn}")
    ax2.set_xlabel("min_cluster_size")
    ax2.set_ylabel("Coerência média")
    ax2.set_title("(b) Efeito do tamanho mínimo do grupo")
    ax2.legend(fontsize=8)
    return _salvar(fig, "grid_hiperparametros.png")


# ---------------------------------------------------------------------------
# 2. Tamanho dos topicos
# ---------------------------------------------------------------------------
def fig_tamanho_topicos(resumo: pd.DataFrame, max_topicos: int = 20) -> Path | None:
    if resumo.empty:
        return None
    t = resumo.copy()
    outliers = t[t["topico"] == -1]
    t = t[t["topico"] != -1].sort_values("n_anuncios", ascending=False).head(max_topicos)
    t = pd.concat([t, outliers]).iloc[::-1]
    rotulos = [("Outliers" if r.topico == -1 else f"T{int(r.topico)}: {_curto(r.palavras_chave)}")
               for r in t.itertuples()]
    cores = [COR_OUTLIER if r.topico == -1 else COR for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(t) + 1.6))
    ax.barh(rotulos, t["pct"], color=cores)
    for y, (pct, n) in enumerate(zip(t["pct"], t["n_anuncios"])):
        ax.text(pct + 0.2, y, f"{pct:.1f}% (n={int(n)})", va="center", fontsize=8)
    ax.set_xlabel("% dos anúncios")
    ax.set_title("Tamanho de cada tópico")
    ax.set_xlim(0, max(t["pct"]) * 1.25)
    return _salvar(fig, "tamanho_topicos.png")


# ---------------------------------------------------------------------------
# 3. Mapa 2D
# ---------------------------------------------------------------------------
def fig_mapa_topicos(anuncios: pd.DataFrame, embeddings: np.ndarray, resumo: pd.DataFrame,
                     n_legenda: int = 12) -> Path | None:
    if len(anuncios) != len(embeddings) or len(anuncios) < 10:
        return None
    from src.modelagem import criar_umap
    umap2d = criar_umap(len(anuncios), n_neighbors=15, n_components=2, min_dist=0.1)
    xy = umap2d.fit_transform(embeddings)

    topicos = anuncios["topico"].to_numpy()
    maiores = (anuncios[anuncios["topico"] != -1]["topico"].value_counts().head(n_legenda).index.tolist())
    cor_de = dict(zip(maiores, _cores(len(maiores))))
    palavras = dict(zip(resumo["topico"].astype(int), resumo["palavras_chave"].astype(str)))

    fig, ax = plt.subplots(figsize=(10.5, 7))
    mask = topicos == -1
    if mask.any():
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, c=COR_OUTLIER, marker="x", linewidth=0.8, label="Outliers")
    outros = ~mask & ~np.isin(topicos, maiores)
    if outros.any():
        ax.scatter(xy[outros, 0], xy[outros, 1], s=14, c="#D8D8D8", label="Outros tópicos")
    for t in maiores:
        m = topicos == t
        ax.scatter(xy[m, 0], xy[m, 1], s=18, color=cor_de[t], label=f"T{t}: {_curto(palavras.get(t, ''), 3)}")
        cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        ax.text(cx, cy, str(t), fontsize=9, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Mapa dos anúncios (cada ponto é um anúncio)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, markerscale=1.4)
    return _salvar(fig, "mapa_topicos.png")


# ---------------------------------------------------------------------------
# 4-7. Figuras que dependem da anotacao
# ---------------------------------------------------------------------------
def _ordem_categorias(serie: pd.Series) -> list[str]:
    cats = serie.value_counts().index.tolist()
    return [c for c in cats if c != SEM_ROTULO] + ([SEM_ROTULO] if SEM_ROTULO in cats else [])


def fig_prevalencia(rotulados: pd.DataFrame) -> Path | None:
    if rotulados.empty:
        return None
    cats = _ordem_categorias(rotulados["categoria"])
    unicos = rotulados["categoria"].value_counts(normalize=True).reindex(cats).fillna(0) * 100
    tem_copias = "n_copias_grupo" in rotulados.columns
    pond = None
    if tem_copias:
        soma = rotulados.groupby("categoria")["n_copias_grupo"].sum().reindex(cats).fillna(0)
        pond = 100 * soma / soma.sum()

    y = np.arange(len(cats))
    h = 0.38 if tem_copias else 0.6
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(cats) + 1.8))
    b1 = ax.barh(y - (h / 2 if tem_copias else 0), unicos.values, h, color=COR, label="Anúncios únicos")
    ax.bar_label(b1, labels=[f"{v:.1f}%" for v in unicos.values], padding=3, fontsize=8)
    if tem_copias:
        b2 = ax.barh(y + h / 2, pond.values, h, color="#E39A3B", label="Contando as cópias")
        ax.bar_label(b2, labels=[f"{v:.1f}%" for v in pond.values], padding=3, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel("% dos anúncios")
    ax.set_title("Prevalência por categoria de risco")
    ax.set_xlim(0, max(unicos.max(), pond.max() if pond is not None else 0) * 1.2 + 1)
    ax.legend(loc="lower right")
    return _salvar(fig, "prevalencia_categorias.png")


def fig_categorias_no_tempo(rotulados: pd.DataFrame) -> Path | None:
    if "mes" not in rotulados.columns:
        return None
    r = rotulados[rotulados["mes"].fillna("").astype(str) != ""]
    if r.empty:
        return None
    tabela = r.groupby(["mes", "categoria"]).size().unstack(fill_value=0).sort_index()
    tabela = tabela[_ordem_categorias(r["categoria"])]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    base = np.zeros(len(tabela))
    for cat, cor in zip(tabela.columns, _cores(len(tabela.columns))):
        ax.bar(tabela.index, tabela[cat].values, bottom=base, label=cat, color=cor)
        base += tabela[cat].values
    passo = max(1, len(tabela) // 12)
    ax.set_xticks(range(0, len(tabela), passo))
    ax.set_xticklabels(tabela.index[::passo], rotation=45, ha="right")
    ax.set_ylabel("Número de anúncios")
    ax.set_xlabel("Mês")
    ax.set_title("Anúncios por mês e categoria de risco")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    return _salvar(fig, "categorias_no_tempo.png")


def _colunas_taticas(rotulados: pd.DataFrame) -> list[str]:
    return [c for c in COL_TATICAS if c in rotulados.columns]


def fig_taticas(rotulados: pd.DataFrame) -> Path | None:
    cols = _colunas_taticas(rotulados)
    if not cols or not rotulados[cols].to_numpy().any():
        return None
    pct = (rotulados[cols].mean() * 100).sort_values()
    nomes = [EIXO2[c.removeprefix("tatica_")][0] for c in pct.index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(cols) + 1.6))
    barras = ax.barh(nomes, pct.values, color=COR)
    ax.bar_label(barras, labels=[f"{v:.1f}%" for v in pct.values], padding=3, fontsize=8)
    ax.set_xlabel("% dos anúncios com a tática")
    ax.set_title("Táticas de persuasão mais frequentes")
    ax.set_xlim(0, max(pct.max() * 1.2, 5))
    return _salvar(fig, "taticas.png")


def fig_categoria_x_tatica(rotulados: pd.DataFrame, min_anuncios: int = 5) -> Path | None:
    cols = _colunas_taticas(rotulados)
    if not cols or not rotulados[cols].to_numpy().any():
        return None
    contagem = rotulados["categoria"].value_counts()
    cats = [c for c in _ordem_categorias(rotulados["categoria"]) if contagem[c] >= min_anuncios]
    if not cats:
        return None
    m = np.array([[100 * rotulados.loc[rotulados["categoria"] == c, t].mean() for t in cols] for c in cats])
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3.5, 0.7 * len(cats) + 2.2))
    img = ax.imshow(m, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([EIXO2[c.removeprefix("tatica_")][0] for c in cols], rotation=40, ha="right")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{c} (n={contagem[c]})" for c in cats])
    for i in range(len(cats)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 55 else "black")
    fig.colorbar(img, ax=ax, label="% dos anúncios da categoria")
    ax.set_title("Táticas por categoria de risco")
    return _salvar(fig, "categoria_x_tatica.png")


def nuvens_por_categoria(rotulados: pd.DataFrame, min_anuncios: int = 5) -> list[Path]:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  Pulado: nuvens (instale com: pip install wordcloud)")
        return []
    if "texto_vetor" not in rotulados.columns:
        return []
    pasta = pasta_figuras() / "nuvens"
    pasta.mkdir(exist_ok=True)
    caminhos = []
    for cat, sub in rotulados.groupby("categoria"):
        if len(sub) < min_anuncios:
            continue
        contagem: dict[str, int] = {}
        for texto in sub["texto_vetor"].fillna("").astype(str):
            for palavra in texto.split():
                contagem[palavra] = contagem.get(palavra, 0) + 1
        if not contagem:
            continue
        nuvem = WordCloud(width=1600, height=800, background_color="white", collocations=False,
                          random_state=config.SEED).generate_from_frequencies(contagem)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.imshow(nuvem, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{cat} (n={len(sub)})")
        caminhos.append(_salvar(fig, f"nuvem_{_slug(cat)}.png", pasta))
    return caminhos


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------
def gerar_todas(usar_tfidf: bool = False, com_mapa: bool = True) -> dict:
    global _MARCA
    _MARCA = _texto_marca(usar_tfidf)
    prefixo = prefixo_resultados(usar_tfidf)
    sufixo_grid = "_tfidf_teste" if usar_tfidf else ""
    saidas: dict[str, object] = {}

    def registrar(nome: str, resultado, motivo: str) -> None:
        ok = bool(resultado)
        saidas[nome] = resultado if ok else None
        print(f"  {'OK    ' if ok else 'Pulada'} {nome}" + ("" if ok else f"  ({motivo})"))

    print(f"Salvando em {pasta_figuras()}")

    arq_grid = config.DIR_SAIDA / f"grid_resultados{sufixo_grid}.csv"
    if arq_grid.exists():
        arq_esc = config.DIR_SAIDA / f"grid_escolhido{sufixo_grid}.json"
        escolhido = json.loads(arq_esc.read_text(encoding="utf-8")) if arq_esc.exists() else None
        registrar("grid_hiperparametros", fig_grid(pd.read_csv(arq_grid), escolhido), "tabela vazia")
    else:
        registrar("grid_hiperparametros", None, "rode: python -m src.grid")

    arq_anuncios = config.DIR_SAIDA / f"{prefixo}_anuncios_com_topicos.csv"
    arq_resumo = config.DIR_SAIDA / f"{prefixo}_resumo_topicos.csv"
    if arq_anuncios.exists() and arq_resumo.exists():
        anuncios = pd.read_csv(arq_anuncios, encoding="utf-8")
        resumo = pd.read_csv(arq_resumo, encoding="utf-8-sig")
        registrar("tamanho_topicos", fig_tamanho_topicos(resumo), "sem topicos")
        if com_mapa:
            arq_emb = caminho_embeddings(usar_tfidf)
            emb = np.load(arq_emb) if arq_emb.exists() else None
            registrar("mapa_topicos",
                      fig_mapa_topicos(anuncios, emb, resumo) if emb is not None else None,
                      "embeddings ausentes ou desalinhados com os anuncios")
    else:
        registrar("tamanho_topicos", None, "rode: python -m src.modelagem --dedup")
        registrar("mapa_topicos", None, "rode: python -m src.modelagem --dedup")

    arq_rot = caminho_rotulados(usar_tfidf)
    if arq_rot.exists():
        rot = pd.read_csv(arq_rot, encoding="utf-8")
        for col in COL_TATICAS:
            if col in rot.columns:
                rot[col] = rot[col].astype(bool)
        registrar("prevalencia_categorias", fig_prevalencia(rot), "sem dados")
        registrar("categorias_no_tempo", fig_categorias_no_tempo(rot), "sem datas validas")
        registrar("taticas", fig_taticas(rot), "nenhuma tatica marcada na aba Topicos")
        registrar("categoria_x_tatica", fig_categoria_x_tatica(rot), "nenhuma tatica marcada")
        registrar("nuvens_por_categoria", nuvens_por_categoria(rot), "sem categorias com anuncios suficientes")
    else:
        for nome in ("prevalencia_categorias", "categorias_no_tempo", "taticas",
                     "categoria_x_tatica", "nuvens_por_categoria"):
            registrar(nome, None, "preencha a anotacao e rode: python -m src.anotacao aplicar")
    return saidas


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera as figuras finais do relatorio.")
    ap.add_argument("--tfidf", action="store_true", help="usa os resultados do teste TF-IDF")
    ap.add_argument("--sem-mapa", action="store_true", help="pula o mapa 2D (mais rapido)")
    args = ap.parse_args()
    modo = "DADOS INVENTADOS (mock): figuras carimbadas" if config.USAR_MOCK else "BASE REAL"
    if args.tfidf and not config.USAR_MOCK:
        modo += " com VETORES DE TESTE: figuras carimbadas"
    print(f"Modo: {modo}")
    gerar_todas(usar_tfidf=args.tfidf, com_mapa=not args.sem_mapa)


if __name__ == "__main__":
    executar(main)
