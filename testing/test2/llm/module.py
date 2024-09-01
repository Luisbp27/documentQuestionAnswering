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
    """Añadir etiquetas de valor en la parte superior de cada barra."""
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge")