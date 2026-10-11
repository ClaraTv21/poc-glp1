"""
Configuracoes centrais da POC (anuncios de GLP-1 na Meta).

Tudo o que pode mudar quando o banco real chegar fica AQUI.
Os outros arquivos so leem estes valores.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# INTERRUPTOR PRINCIPAL
#   True  -> usa os dados INVENTADOS de data/mock (so para testar o codigo)
#   False -> usa a base REAL em data/raw
# Resultados de teste e resultados reais ficam em pastas separadas
# (outputs/mock e outputs/real), entao nada se mistura.
# ---------------------------------------------------------------------------
USAR_MOCK = True

DIR_RAW = RAIZ / "data" / "raw"
DIR_MOCK = RAIZ / "data" / "mock"
DIR_CLEAN = RAIZ / "data" / "clean"

_SUFIXO = "mock" if USAR_MOCK else "real"

# Arquivo de entrada. Aceita .csv, .tsv, .parquet, .json/.jsonl e SQLite (.db/.sqlite)
ARQ_DADOS = (DIR_MOCK / "anuncios_teste.csv") if USAR_MOCK else (DIR_RAW / "anuncios.csv")

# Se a entrada for um banco SQLite com varias tabelas, escreva aqui o nome da
# tabela de anuncios (ou deixe None para o codigo listar as tabelas para voce).
TABELA_SQLITE = None

# Nomes das colunas NO SEU ARQUIVO (troque quando souber os nomes reais).
COL_ID = "id"
COL_TEXTO = "texto"
COL_ANUNCIANTE = "anunciante"
COL_DATA = "data"

# Datas escritas como 03/02/2025: True = 3 de fevereiro (padrao brasileiro),
# False = 2 de marco. Datas no formato 2025-02-03 nao sao afetadas.
DATA_DIA_PRIMEIRO = True

# Arquivos de saida
ARQ_LIMPO = DIR_CLEAN / f"anuncios_limpos_{_SUFIXO}.csv"
DIR_SAIDA = RAIZ / "outputs" / _SUFIXO
ARQ_EMBEDDINGS = DIR_SAIDA / "embeddings.npy"

# Reprodutibilidade: use SEMPRE este valor nos modelos
SEED = 42

# Modelo de embeddings (multilingue, funciona bem em portugues)
MODELO_EMBEDDING = "paraphrase-multilingual-mpnet-base-v2"

# Limpeza: textos limpos com menos caracteres que isto sao descartados
MIN_CARACTERES = 15

# Quase-duplicatas (simhash): distancia de Hamming maxima (de 64 bits) e
# similaridade minima (Jaccard de palavras) para confirmar o par.
# Padrao CONSERVADOR (k=3, J=0.8): junta so variacoes pequenas (um numero, uma
# palavra). Com k=6 e J=0.6 o codigo passa a juntar "modelos de anuncio" inteiros
# (mesma campanha com nome de produto/preco trocados): util como analise a parte.
DISTANCIA_SIMHASH = 3
JACCARD_MINIMO = 0.8


def garantir_pastas() -> None:
    """Cria as pastas de trabalho se ainda nao existirem."""
    for pasta in (DIR_RAW, DIR_MOCK, DIR_CLEAN, DIR_SAIDA):
        pasta.mkdir(parents=True, exist_ok=True)
