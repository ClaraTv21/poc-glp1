from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Frases INVENTADAS apenas para testar o pipeline (nao sao dados reais)
docs = [
    # tema A: redirecionamento para WhatsApp
    "Chame no WhatsApp e garanta sua caneta",
    "Clique e fale com nosso atendente no zap agora",
    "Peça pelo WhatsApp com entrega rápida",
    "Chama no direct ou zap para fazer seu pedido",
    "Link na bio, atendimento pelo WhatsApp",
    "Mande mensagem no WhatsApp e receba hoje",
    "Fale com a gente no zap e peça sua caneta",
    "Atendimento exclusivo pelo WhatsApp, chame já",
    # tema B: promessas de emagrecimento
    "Emagreça 10 kg em um mês sem dieta",
    "Perca peso rápido com a caneta emagrecedora",
    "Resultado garantido: elimine a barriga em 30 dias",
    "Seque 15 quilos sem esforço e sem academia",
    "Fórmula milagrosa derrete a gordura abdominal",
    "Menos 8 kg em 3 semanas, resultado comprovado",
    "Emagrecimento rápido e definitivo com o novo tratamento",
    "Perca medidas em poucos dias sem efeito sanfona",
    # tema C: preco e Pix
    "Ozempic por apenas R$ 299 pagando no Pix",
    "Mounjaro com 70% de desconto só hoje no Pix",
    "Promoção relâmpago: pague no Pix e leve 2 canetas",
    "Preço imbatível, aceitamos Pix e cartão",
    "Frete grátis e desconto especial no Pix",
    "Oferta limitada: R$ 399 a caneta, pague via Pix",
    "Pague só no Pix e ganhe desconto extra na compra",
    "Últimas unidades com preço promocional no Pix",
]

embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

modelo = BERTopic(
    embedding_model=embedder,
    umap_model=UMAP(n_neighbors=5, n_components=2, min_dist=0.0,
                    metric="cosine", random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=3, metric="euclidean",
                          cluster_selection_method="eom",
                          prediction_data=True),
    vectorizer_model=CountVectorizer(ngram_range=(1, 2)),
    language="multilingual",
    verbose=True,
)

topicos, probs = modelo.fit_transform(docs)
print(modelo.get_topic_info())