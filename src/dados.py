"""
Leitura, auditoria e preparacao dos anuncios.

Fluxo:
    carregar_dados -> padronizar_colunas -> auditar -> preparar_dados -> salvar_limpo

Atalho: obter_dados_limpos() faz tudo isso (e reaproveita o arquivo limpo se
ele ja existir e estiver atualizado).

Uso pelo terminal (na raiz do projeto):
    python -m src.dados            # audita, limpa e salva
    python -m src.dados --forcar   # refaz mesmo que o arquivo limpo ja exista
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

from src import config
from src.cli import executar  # noqa: E402
from src.limpeza import contar_emojis, limpar_para_embedding, limpar_para_vetorizacao

COLUNAS_PADRAO = ["id", "texto", "anunciante", "data"]


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def _detectar_separador(caminho: Path, encoding: str) -> str:
    """Descobre o separador pelo cabecalho (a 1a linha): o mais frequente entre , ; tab |."""
    with open(caminho, encoding=encoding, newline="") as f:
        cabecalho = f.readline()
    contagem = {sep: cabecalho.count(sep) for sep in (",", ";", "\t", "|")}
    melhor = max(contagem, key=contagem.get)
    return melhor if contagem[melhor] > 0 else ","


def _ler_csv(caminho: Path, sep: str | None = None) -> pd.DataFrame:
    """
    Le CSV/TSV. Detecta o separador (CSV exportado em pt-BR costuma usar ';') e
    tenta UTF-8 e, se falhar, Latin-1.
    """
    for enc in ("utf-8-sig", "latin-1"):
        try:
            separador = sep or _detectar_separador(caminho, enc)
            return pd.read_csv(caminho, sep=separador, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Nao consegui ler {caminho} (codificacao desconhecida).")


def _ler_sqlite(caminho: Path, tabela: str | None) -> pd.DataFrame:
    con = sqlite3.connect(caminho.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            tabelas = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%'")]
        except sqlite3.DatabaseError as erro:
            raise ValueError(
                "Nao consegui abrir o banco. Se ele pede senha no DB Browser, "
                "esta criptografado (SQLCipher) e o Python comum nao le. "
                "Abra no DB Browser com a senha e exporte a tabela de anuncios "
                "para CSV (Arquivo > Exportar > Tabela(s) como CSV)."
            ) from erro

        tabela = tabela or config.TABELA_SQLITE
        if tabela is None:
            if len(tabelas) == 1:
                tabela = tabelas[0]
            else:
                detalhes = []
                for t in tabelas:
                    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
                    detalhes.append(f"  - {t}: {cols}")
                raise ValueError(
                    "O banco tem varias tabelas. Escolha uma e escreva o nome em "
                    "config.TABELA_SQLITE:\n" + "\n".join(detalhes))
        if tabela not in tabelas:
            raise ValueError(f"Tabela '{tabela}' nao existe. Tabelas: {tabelas}")
        return pd.read_sql_query(f'SELECT * FROM "{tabela}"', con)
    finally:
        con.close()


def carregar_dados(caminho: str | Path | None = None, tabela: str | None = None) -> pd.DataFrame:
    """Le o arquivo de anuncios (csv, tsv, parquet, json/jsonl ou SQLite)."""
    caminho = Path(caminho or config.ARQ_DADOS)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {caminho}\n"
            "Dica: confira config.ARQ_DADOS e o interruptor config.USAR_MOCK. "
            "Para dados de teste rode: python scripts/gerar_dados_teste.py")
    sufixo = caminho.suffix.lower()
    if sufixo == ".csv":
        return _ler_csv(caminho)
    if sufixo == ".tsv":
        return _ler_csv(caminho, "\t")
    if sufixo == ".parquet":
        return pd.read_parquet(caminho)
    if sufixo in (".json", ".jsonl"):
        return pd.read_json(caminho, lines=(sufixo == ".jsonl"))
    if sufixo in (".db", ".sqlite", ".sqlite3"):
        return _ler_sqlite(caminho, tabela)
    raise ValueError(f"Formato nao suportado: {sufixo}")


# ---------------------------------------------------------------------------
# Padronizacao e auditoria
# ---------------------------------------------------------------------------
def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia as colunas do arquivo (config.COL_*) para os nomes padrao do projeto."""
    mapa_cfg = {
        config.COL_ID: "id",
        config.COL_TEXTO: "texto",
        config.COL_ANUNCIANTE: "anunciante",
        config.COL_DATA: "data",
    }
    if config.COL_TEXTO not in df.columns:
        raise ValueError(
            f"A coluna de texto '{config.COL_TEXTO}' nao existe no arquivo.\n"
            f"Colunas disponiveis: {list(df.columns)}\n"
            "Edite config.COL_TEXTO (e as outras COL_*) com os nomes reais.")

    df = df.copy()
    renomear = {orig: dest for orig, dest in mapa_cfg.items()
                if orig in df.columns and orig != dest}
    # evita duas colunas com o mesmo nome apos renomear
    for orig, dest in renomear.items():
        if dest in df.columns:
            warnings.warn(f"Coluna '{dest}' ja existia e foi substituida por '{orig}'.")
            df = df.drop(columns=[dest])
    df = df.rename(columns=renomear)

    if "id" not in df.columns:
        warnings.warn("Coluna de id nao encontrada: criei ids sequenciais.")
        df.insert(0, "id", range(1, len(df) + 1))
    for opcional in ("anunciante", "data"):
        if opcional not in df.columns:
            warnings.warn(f"Coluna '{opcional}' nao encontrada: algumas analises serao puladas.")
            df[opcional] = pd.NA
    return df.reset_index(drop=True)


def normalizar_chave(texto: str) -> str:
    """Chave para comparar textos: minusculas, sem acento nem pontuacao."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def auditar(df: pd.DataFrame, min_caracteres: int = config.MIN_CARACTERES) -> dict:
    """Retrato da base ANTES da limpeza (espera colunas ja padronizadas)."""
    texto = df["texto"].fillna("").astype(str)
    datas = _converter_datas(df["data"])
    chaves = texto.map(normalizar_chave)
    return {
        "n_linhas": int(len(df)),
        "colunas": list(df.columns),
        "nulos_por_coluna": {c: int(v) for c, v in df.isna().sum().items()},
        "textos_vazios": int((texto.str.strip() == "").sum()),
        "textos_curtos": int(((texto.str.strip() != "") & (texto.str.len() < min_caracteres)).sum()),
        "duplicatas_exatas_texto": int(chaves[chaves != ""].duplicated().sum()),
        "n_anunciantes": int(df["anunciante"].nunique(dropna=True)),
        "datas_invalidas": int(datas.isna().sum() - df["data"].isna().sum()),
        "periodo": (str(datas.min().date()) if datas.notna().any() else None,
                    str(datas.max().date()) if datas.notna().any() else None),
        "chars_media": round(float(texto.str.len().mean()), 1) if len(df) else 0.0,
        "chars_mediana": float(texto.str.len().median()) if len(df) else 0.0,
    }


def imprimir_auditoria(a: dict) -> None:
    print("=" * 60)
    print("AUDITORIA DA BASE (antes da limpeza)")
    print("=" * 60)
    print(f"Linhas: {a['n_linhas']}   Anunciantes distintos: {a['n_anunciantes']}")
    print(f"Periodo: {a['periodo'][0]} a {a['periodo'][1]}   Datas invalidas: {a['datas_invalidas']}")
    print(f"Textos vazios: {a['textos_vazios']}   Textos curtos: {a['textos_curtos']}")
    print(f"Duplicatas exatas de texto: {a['duplicatas_exatas_texto']}")
    print(f"Tamanho do texto (caracteres): media {a['chars_media']}, mediana {a['chars_mediana']}")
    nulos = {k: v for k, v in a["nulos_por_coluna"].items() if v}
    print(f"Nulos por coluna: {nulos if nulos else 'nenhum'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Preparacao
# ---------------------------------------------------------------------------
def _converter_datas(serie: pd.Series) -> pd.Series:
    """
    Converte para data; valores invalidos viram NaT (nao quebra o codigo).
    - Datas no formato ISO (2025-02-03, com ou sem hora) sao lidas como ano-mes-dia.
    - As demais (ex.: 03/02/2025) seguem config.DATA_DIA_PRIMEIRO.
    """
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie).dt.tz_localize(None) if getattr(serie.dt, "tz", None) else serie

    texto = serie.astype("string").str.strip()
    eh_iso = texto.str.match(r"^\d{4}-\d{2}-\d{2}").fillna(False).astype(bool)
    eh_outro = (~eh_iso) & texto.notna()
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")

    if eh_iso.any():
        try:
            iso = pd.to_datetime(texto[eh_iso], errors="coerce", format="ISO8601", utc=True)
        except (ValueError, TypeError):
            iso = pd.to_datetime(texto[eh_iso], errors="coerce", utc=True)
        resultado[eh_iso] = iso.dt.tz_localize(None).astype("datetime64[ns]")
    if eh_outro.any():
        dia_primeiro = config.DATA_DIA_PRIMEIRO
        try:
            outros = pd.to_datetime(texto[eh_outro], errors="coerce", format="mixed",
                                    dayfirst=dia_primeiro)
        except (ValueError, TypeError):
            outros = pd.to_datetime(texto[eh_outro], errors="coerce", dayfirst=dia_primeiro)
        resultado[eh_outro] = outros.astype("datetime64[ns]")
    return resultado


def preparar_dados(df: pd.DataFrame, min_caracteres: int = config.MIN_CARACTERES,
                   verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Limpa os textos, remove anuncios vazios/curtos e calcula colunas de apoio.

    Colunas criadas:
      texto_limpo   limpeza leve (vai para os embeddings)
      texto_vetor   minusculas e sem stopwords (nuvens de palavras, c-TF-IDF)
      texto_chave   texto normalizado para comparar duplicatas
      n_emojis, n_chars, n_palavras
      n_copias                  quantas vezes o mesmo texto (normalizado) aparece
      n_anunciantes_mesmo_texto quantos anunciantes usam exatamente esse texto
      dup_exata                 True a partir da 2a ocorrencia do mesmo texto
      mes                       'AAAA-MM' (vazio se a data for invalida)
    """
    df = df.copy()
    n_inicial = len(df)

    if df["id"].duplicated().any():
        warnings.warn("Ha ids repetidos na base; confira a coluna de id.")

    bruto = df["texto"].fillna("").astype(str)
    df["texto_limpo"] = bruto.map(limpar_para_embedding)
    df["n_emojis"] = bruto.map(contar_emojis)

    tem_letra = df["texto_limpo"].str.contains(r"[^\W\d_]", regex=True)
    comprimento_ok = df["texto_limpo"].str.len() >= min_caracteres
    valido = tem_letra & comprimento_ok
    n_removidos = int((~valido).sum())
    df = df[valido].reset_index(drop=True)

    df["texto_vetor"] = df["texto_limpo"].map(limpar_para_vetorizacao)
    df["n_chars"] = df["texto_limpo"].str.len()
    df["n_palavras"] = df["texto_limpo"].str.split().str.len()

    df["texto_chave"] = df["texto_limpo"].map(normalizar_chave)
    grupos = df.groupby("texto_chave")
    df["n_copias"] = grupos["texto_chave"].transform("size").astype(int)
    df["n_anunciantes_mesmo_texto"] = grupos["anunciante"].transform(
        lambda s: s.nunique(dropna=True)).astype(int)
    df["dup_exata"] = df["texto_chave"].duplicated(keep="first")

    datas = _converter_datas(df["data"])
    df["data"] = datas
    df["mes"] = datas.dt.strftime("%Y-%m").where(datas.notna(), other="")

    relatorio = {
        "n_inicial": n_inicial,
        "n_removidos_vazios_ou_curtos": n_removidos,
        "n_final": int(len(df)),
        "n_textos_unicos": int(df["texto_chave"].nunique()),
        "n_duplicatas_exatas": int(df["dup_exata"].sum()),
    }
    if verbose:
        print(f"Limpeza: {n_inicial} -> {relatorio['n_final']} anuncios "
              f"({n_removidos} removidos por vazios/curtos/sem letras). "
              f"Textos unicos: {relatorio['n_textos_unicos']}.")
    return df, relatorio


def salvar_limpo(df: pd.DataFrame, caminho: str | Path | None = None) -> Path:
    caminho = Path(caminho or config.ARQ_LIMPO)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def ler_limpo(caminho: str | Path | None = None) -> pd.DataFrame:
    """Le o arquivo limpo salvo por salvar_limpo (restaura tipos)."""
    caminho = Path(caminho or config.ARQ_LIMPO)
    df = pd.read_csv(caminho, encoding="utf-8")
    for col in ("texto", "texto_limpo", "texto_vetor", "texto_chave", "mes"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["data"] = _converter_datas(df["data"])
    df["dup_exata"] = df["dup_exata"].astype(bool)
    return df


def obter_dados_limpos(forcar: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Devolve os dados limpos. Reaproveita o arquivo limpo se ele existir e for
    mais novo que o arquivo de entrada; caso contrario refaz tudo.
    """
    config.garantir_pastas()
    entrada, limpo = Path(config.ARQ_DADOS), Path(config.ARQ_LIMPO)
    if (not forcar and limpo.exists() and entrada.exists()
            and limpo.stat().st_mtime >= entrada.stat().st_mtime):
        if verbose:
            print(f"Usando dados limpos ja salvos: {limpo}")
        return ler_limpo(limpo)

    bruto = padronizar_colunas(carregar_dados())
    if verbose:
        imprimir_auditoria(auditar(bruto))
    df, _ = preparar_dados(bruto, verbose=verbose)
    salvar_limpo(df)
    if verbose:
        print(f"Dados limpos salvos em {limpo}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita e limpa a base de anuncios.")
    ap.add_argument("--forcar", action="store_true", help="refaz mesmo que ja exista")
    args = ap.parse_args()
    modo = "DADOS INVENTADOS (mock)" if config.USAR_MOCK else "BASE REAL"
    print(f"Modo: {modo} | entrada: {config.ARQ_DADOS}")
    df = obter_dados_limpos(forcar=args.forcar)
    print(df[["id", "anunciante", "data", "texto_limpo", "n_copias"]].head(5).to_string())


if __name__ == "__main__":
    executar(main)
