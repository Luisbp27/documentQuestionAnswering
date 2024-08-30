import pandas as pd

def get_df(df_cat_mean, df_en_mean, df_es_mean, mode):
    if mode == "gen":
        df_means = pd.DataFrame({
            "Metric": ["Catalán", "Inglés", "Español"],
            "Faithfulness": [df_cat_mean["faithfulness"], df_en_mean["faithfulness"], df_es_mean["faithfulness"]],
            "Answer Relevancy": [df_cat_mean["answer_relevancy"], df_en_mean["answer_relevancy"], df_es_mean["answer_relevancy"]]
        }).set_index("Metric").T
    elif mode == "ret":
        df_means = pd.DataFrame({
            "Metric": ["Catalán", "Inglés", "Español"],
            "Context Precision": [df_cat_mean["context_precision"], df_en_mean["context_precision"], df_es_mean["context_precision"]],
            "Context Recall": [df_cat_mean["context_recall"], df_en_mean["context_recall"], df_es_mean["context_recall"]]
        }).set_index("Metric").T
    else:
        raise ValueError("Invalid mode")

    return df_means

# Función para añadir valores en las barras
def add_values_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')