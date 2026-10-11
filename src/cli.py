"""Ajuda comum aos comandos de terminal (python -m src.xxx)."""
from __future__ import annotations

import sys


def executar(funcao) -> None:
    """
    Roda a funcao main() e, se acontecer um erro PREVISTO (arquivo ausente,
    coluna errada, pacote nao instalado), mostra so a mensagem, sem o
    'Traceback' comprido. Erros inesperados continuam aparecendo completos.
    """
    try:
        funcao()
    except (FileNotFoundError, ValueError, ImportError) as erro:
        print(f"\nERRO: {erro}", file=sys.stderr)
        raise SystemExit(1) from None
