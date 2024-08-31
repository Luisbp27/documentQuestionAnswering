from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

def get_df(df_cat_mean, df_en_mean, df_es_mean):
    df_means = pd.DataFrame({
        "Idioma": ["Catalán", "Inglés", "Español"],
        "Faithfulness": [df_cat_mean["faithfulness"], df_en_mean["faithfulness"], df_es_mean["faithfulness"]],
        "Answer Relevancy": [df_cat_mean["answer_relevancy"], df_en_mean["answer_relevancy"], df_es_mean["answer_relevancy"]]
    }).set_index("Idioma").T

    return df_means

# Función para añadir valores en las barras
def add_values_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')