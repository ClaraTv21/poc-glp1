src/config.py

scripts/gerar_dados_teste.py

que cria data/mock/anuncios_teste.csv com 300 a 500 anúncios inventados em português. Tem que parecer a realidade:

6 narrativas, por exemplo: preço baixo e Pix; "sem receita"; WhatsApp; resultado rápido; falso frete; informação médica legítima.
Sujeira de propósito: HTML (<b>), URLs, @menções, emojis, "Patrocinado", "Saiba mais".
Ofuscação: "ozempic" escrito de outras formas ("O z e m p i c", "ozenpic", "0zempic").
Duplicatas e quase-duplicatas (mesmo anúncio com uma palavra trocada).
Colunas iguais às que você espera: id, anunciante, data, texto.

duplicatas.py	Dedupe exato + quase-duplicatas (simhash)	Agrupa os anúncios que você duplicou de propósito
descritiva.py	Estatísticas: tamanho, palavras frequentes, por mês e por anunciante	Gera tabelas e gráficos em outputs/
embeddings.py	Calcula e salva em disco os embeddings	Gera outputs/embeddings.npy; na 2ª vez carrega sem recalcular
modelagem.py	Função que monta o BERTopic com os parâmetros recebidos	Acha ~6 tópicos nos dados de teste
avaliacao.py	Mede cada modelo: nº de tópicos, % de outliers, coerência, diversidade	Devolve uma linha de tabela por modelo
grid.py	Roda várias combinações e grava tudo em outputs/grid.csv	Roda um mini-grid de 4 combinações
anotacao.py	Sorteia anúncios de cada tópico e gera a planilha de anotação	Planilha com 15 anúncios por tópico
figuras.py	Gráficos finais (barras, linha do tempo, nuvens de palavras)	Gera os PNGs com os dados de teste