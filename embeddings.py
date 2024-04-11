
from transformers import AutoTokenizer, AutoModel
import torch

# Cargar el tokenizer y el modelo
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

def get_embeddings(text_chunks):
    embeddings = []
    for text in text_chunks:
        # Procesar el texto con el tokenizer y crear tensores PyTorch
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Obtener embeddings del modelo
        with torch.no_grad():
            outputs = model(**inputs)

        # Usar embeddings del último estado oculto
        # Aquí usamos mean pooling para obtener un único vector por fragmento
        last_hidden_states = outputs.last_hidden_state
        mean_embedding = torch.mean(last_hidden_states, dim=1)
        embeddings.append(mean_embedding)

    # Convertir la lista de tensores en un tensor
    embeddings_tensor = torch.cat(embeddings, dim=0)
    return embeddings_tensor

# Asumiendo que `text_chunks` es la lista de fragmentos de texto obtenida previamente
# text_chunks = split_text(documentos)

# Obtener embeddings
# embeddings = get_embeddings(text_chunks)
