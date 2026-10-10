from sentence_transformers import SentenceTransformer

nome = "paraphrase-multilingual-mpnet-base-v2"
print("Baixando/carregando modelo:", nome)
modelo = SentenceTransformer(nome)
print("Modelo carregado com sucesso.")

frases = [
    "Emagreça 10kg em um mês com Ozempic original",
    "Chame no WhatsApp e garanta sua caneta com desconto",
]
emb = modelo.encode(frases, show_progress_bar=True)
print("Formato dos embeddings:", emb.shape)