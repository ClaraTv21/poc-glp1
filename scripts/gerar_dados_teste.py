"""
Gera um CSV com anuncios INVENTADOS para testar o pipeline.

ATENCAO: nada aqui e dado real. Estes dados servem so para verificar se o
codigo funciona. NUNCA use resultados deles no relatorio.

O arquivo tem sujeira de proposito (HTML, URL, @mencao, emojis, "Patrocinado",
"Saiba mais"), grafias ofuscadas ("O z e m p i c"), duplicatas exatas,
quase-duplicatas, textos vazios/curtos e datas invalidas.

Uso (na raiz do projeto):
    python scripts/gerar_dados_teste.py
    python scripts/gerar_dados_teste.py --n 500 --seed 7
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# permite "from src import config" quando rodado como script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402

PRODUTOS = ["Ozempic", "Mounjaro", "Wegovy", "Saxenda", "semaglutida", "tirzepatide"]

# narrativa -> (peso, modelos de frase)
NARRATIVAS = {
    "preco_pix": (0.22, [
        "{prod} por apenas R$ {preco} pagando no Pix! Promoção só hoje.",
        "Oferta imperdível: {prod} com {desc}% de desconto no Pix. Garanta o seu!",
        "Preço de fábrica! {prod} a partir de R$ {preco}. Aceitamos Pix e cartão.",
        "Queima de estoque: {prod} R$ {preco} no Pix, frete grátis para todo o Brasil.",
    ]),
    "sem_receita": (0.15, [
        "{prod} sem receita médica, sem burocracia. Peça agora e receba em casa.",
        "Quer emagrecer? {prod} original sem precisar de receita. Entrega discreta.",
        "Compre {prod} sem consulta e sem receita. Envio rápido e embalagem discreta.",
    ]),
    "whatsapp": (0.20, [
        "Chame no WhatsApp {tel} e garanta sua caneta {prod}. Atendimento imediato!",
        "Fale com nosso atendente pelo zap {tel} e peça o seu {prod}.",
        "Link na bio! Clique e chame no WhatsApp para comprar {prod} com desconto.",
        "Atendimento exclusivo pelo WhatsApp {tel}. Peça seu {prod} hoje mesmo.",
    ]),
    "resultado_rapido": (0.18, [
        "Perca {kg}kg em {dias} dias com {prod}. Resultado garantido, sem dieta e sem academia!",
        "Elimine a barriga em {dias} dias! {prod}, a caneta emagrecedora que derrete a gordura.",
        "Menos {kg}kg em {dias} dias, resultado comprovado por quem usou. Use {prod}.",
    ]),
    "falso_frete": (0.10, [
        "Seu pedido de {prod} está retido na alfândega. Pague a taxa de liberação de R$ {taxa} via Pix.",
        "Atenção: encomenda {prod} aguardando liberação. Regularize a taxa de R$ {taxa} para receber hoje.",
        "Rastreio: pacote retido! Para liberar a entrega do seu {prod}, pague R$ {taxa} de frete no Pix.",
    ]),
    "informativo": (0.15, [
        "Entenda como a semaglutida age no organismo e quando o tratamento é indicado. Converse com seu médico.",
        "Obesidade é doença: conheça as opções de tratamento com acompanhamento médico e nutricional.",
        "Agende uma consulta com endocrinologista para avaliar o uso de {prod}. Medicamento vendido somente com receita.",
        "Dicas de alimentação e atividade física para quem está em tratamento da obesidade. Fale com seu médico.",
    ]),
}

OFUSCACOES = {
    "Ozempic": ["O z e m p i c", "0zempic", "Ozenpic", "O-zempic", "OZEMP1C"],
    "Mounjaro": ["Mounj@ro", "M0unjaro", "M o u n j a r o", "Mounjar0"],
    "Wegovy": ["W3govy", "We govy"],
}

EMOJIS = ["💉", "🔥", "✅", "😍", "⚠️", "💸", "📲", "🚚"]
SUBSTITUICOES = {"hoje": "agora", "Peça": "Compre", "Garanta": "Aproveite",
                 "rápido": "imediato", "Atenção": "Aviso"}


def _telefone(rng: random.Random) -> str:
    # numero obviamente falso (DDD 99, final 0000)
    return f"(99) 9{rng.randint(1000, 9999)}-0000"


def _preencher(modelo: str, rng: random.Random) -> str:
    return modelo.format(
        prod=rng.choice(PRODUTOS),
        preco=rng.choice([199, 249, 299, 349, 399, 450, 599]),
        desc=rng.choice([30, 40, 50, 60, 70]),
        tel=_telefone(rng),
        kg=rng.choice([5, 8, 10, 12, 15]),
        dias=rng.choice([7, 15, 21, 30, 45]),
        taxa=rng.choice([89, 120, 150, 280, 390]),
    )


def _ofuscar(texto: str, rng: random.Random, prob: float = 0.35) -> str:
    for nome, variantes in OFUSCACOES.items():
        if nome in texto and rng.random() < prob:
            texto = texto.replace(nome, rng.choice(variantes))
    return texto


def _sujar(texto: str, rng: random.Random) -> str:
    """Acrescenta ruido tipico de anuncio coletado."""
    if rng.random() < 0.30:
        texto = "Patrocinado " + texto
    if rng.random() < 0.40:
        texto = texto + " Saiba mais"
    if rng.random() < 0.50:
        texto = rng.choice(EMOJIS) + " " + texto + " " + rng.choice(EMOJIS)
    if rng.random() < 0.15:
        texto = f"<b>{texto}</b>" if rng.random() < 0.5 else texto.replace(". ", ".<br>")
    if rng.random() < 0.25:
        texto += " https://exemplo-teste.invalid/promo?id=" + str(rng.randint(100, 999))
    if rng.random() < 0.10:
        texto += " @loja_teste"
    if rng.random() < 0.15:
        texto += " #emagrecimento"
    return texto


def _quase_duplicata(texto: str, rng: random.Random) -> str:
    """Muda pouco o texto: troca numero, uma palavra ou acrescenta pontuacao."""
    opcao = rng.choice(["numero", "palavra", "pontuacao"])
    if opcao == "numero" and re.search(r"\d+", texto):
        return re.sub(r"\d+", lambda m: str(int(m.group()) + rng.choice([10, 20, 50])),
                      texto, count=1)
    if opcao == "palavra":
        for velha, nova in SUBSTITUICOES.items():
            if velha in texto:
                return texto.replace(velha, nova, 1)
    return texto + "!!"


def gerar(n: int = 450, seed: int = config.SEED, n_anunciantes: int = 80) -> pd.DataFrame:
    rng = random.Random(seed)

    # anunciantes com preferencia por uma narrativa e popularidade desigual
    nomes_narr = list(NARRATIVAS)
    # narrativas distribuidas em rodizio (com peso de popularidade) para equilibrar o teste
    ciclo = [nomes_narr[i % len(nomes_narr)] for i in range(n_anunciantes)]
    rng.shuffle(ciclo)
    anunciantes = []
    for i in range(1, n_anunciantes + 1):
        anunciantes.append({
            "nome": f"anunciante_{i:03d}",
            "narrativa": ciclo[i - 1],
            "peso": 1.0 / (i ** 0.5),
        })
    pesos_anunc = [a["peso"] for a in anunciantes]
    pesos_narr = [NARRATIVAS[k][0] for k in nomes_narr]

    n_base = int(n * 0.84)
    linhas: list[dict] = []
    for _ in range(n_base):
        anunc = rng.choices(anunciantes, weights=pesos_anunc)[0]
        narr = anunc["narrativa"] if rng.random() < 0.85 else rng.choices(nomes_narr, weights=pesos_narr)[0]
        base = _preencher(rng.choice(NARRATIVAS[narr][1]), rng)
        if narr != "informativo":
            base = _ofuscar(base, rng)
        linhas.append({"anunciante": anunc["nome"], "texto_base": base, "narrativa_sintetica": narr})

    # duplicatas exatas (mesmo texto base, as vezes outro anunciante) e quase-duplicatas
    n_dup = int(n * 0.08)
    n_quase = int(n * 0.08)
    for _ in range(n_dup):
        origem = rng.choice(linhas[:n_base])
        outro = rng.choice(anunciantes)["nome"] if rng.random() < 0.4 else origem["anunciante"]
        linhas.append({**origem, "anunciante": outro})
    for _ in range(n_quase):
        origem = rng.choice(linhas[:n_base])
        linhas.append({**origem, "texto_base": _quase_duplicata(origem["texto_base"], rng)})

    registros = []
    for l in linhas:
        registros.append({
            "anunciante": l["anunciante"],
            "texto": _sujar(l["texto_base"], rng),   # cada copia recebe ruido proprio
            "narrativa_sintetica": l["narrativa_sintetica"],
        })

    # casos de borda: vazios, nulos e textos muito curtos
    registros += [{"anunciante": rng.choice(anunciantes)["nome"], "texto": "", "narrativa_sintetica": "ruido"} for _ in range(3)]
    registros += [{"anunciante": rng.choice(anunciantes)["nome"], "texto": None, "narrativa_sintetica": "ruido"} for _ in range(2)]
    registros += [{"anunciante": rng.choice(anunciantes)["nome"], "texto": t, "narrativa_sintetica": "ruido"}
                  for t in ("Ozempic", "oi", "👍👍")]

    rng.shuffle(registros)

    inicio, fim = date(2025, 1, 1), date(2026, 6, 30)
    dias = (fim - inicio).days
    df = pd.DataFrame(registros)
    df.insert(0, "id", range(1, len(df) + 1))
    datas = [(inicio + timedelta(days=rng.randint(0, dias))).isoformat() for _ in range(len(df))]
    df["data"] = datas
    # duas datas invalidas para testar o tratamento de erro
    for idx in rng.sample(range(len(df)), 2):
        df.loc[idx, "data"] = "sem data"
    df["eh_sintetico"] = True
    return df[["id", "anunciante", "data", "texto", "narrativa_sintetica", "eh_sintetico"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera anuncios INVENTADOS para testes.")
    ap.add_argument("--n", type=int, default=450, help="numero aproximado de anuncios")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--saida", type=Path, default=config.DIR_MOCK / "anuncios_teste.csv")
    args = ap.parse_args()

    config.garantir_pastas()
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df = gerar(n=args.n, seed=args.seed)
    df.to_csv(args.saida, index=False, encoding="utf-8")

    aviso = args.saida.parent / "LEIA-ME_DADOS_INVENTADOS.txt"
    aviso.write_text(
        "Os arquivos desta pasta contem dados INVENTADOS, gerados por "
        "scripts/gerar_dados_teste.py, apenas para testar o codigo.\n"
        "Nao use estes dados nem seus resultados no relatorio.\n",
        encoding="utf-8",
    )
    print(f"Gerados {len(df)} anuncios INVENTADOS em {args.saida}")
    print(df["narrativa_sintetica"].value_counts().to_string())


if __name__ == "__main__":
    main()
