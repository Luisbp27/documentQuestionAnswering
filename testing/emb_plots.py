import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    test_names = ["jurídicos", "financieros", "científico-técnicos"]
    for num_test in range(1, 4):
        df_ml = pd.read_csv(f"./test{num_test}/emb/test{num_test}_llama3_multilingual_large.csv")
        df_bl = pd.read_csv(f"./test{num_test}/emb/test{num_test}_llama3_baai_large.csv")
        df_mxl = pd.read_csv(f"./test{num_test}/emb/test{num_test}_llama3_mxbai_large.csv")
        df_bs = pd.read_csv(f"./test{num_test}/emb/test{num_test}_llama3_baai_small.csv")

        df_ml = df_ml.fillna(0, inplace=False)
        df_bl = df_bl.fillna(0, inplace=False)
        df_mxl = df_mxl.fillna(0, inplace=False)
        df_bs = df_bs.fillna(0, inplace=False)

        df_ml = df_ml.drop(columns=["question", "answer", "contexts", "ground_truth", "faithfulness", "answer_relevancy"])
        df_bl = df_bl.drop(columns=["question", "answer", "contexts", "ground_truth", "faithfulness", "answer_relevancy"])
        df_mxl = df_mxl.drop(columns=["question", "answer", "contexts", "ground_truth", "faithfulness", "answer_relevancy"])
        df_bs = df_bs.drop(columns=["question", "answer", "contexts", "ground_truth", "faithfulness", "answer_relevancy"])

        df_ml_mean = df_ml.mean()
        df_bl_mean = df_bl.mean()
        df_mxl_mean = df_mxl.mean()
        df_bs_mean = df_bs.mean()

        df_ml_mean = df_ml_mean.to_frame(name="Mulilingual Large")
        df_bl_mean = df_bl_mean.to_frame(name="BAAI Large")
        df_mxl_mean = df_mxl_mean.to_frame(name="MXBAI Large")
        df_bs_mean = df_bs_mean.to_frame(name="BAAI Small")

        df_combined = pd.concat([df_ml_mean, df_bl_mean, df_mxl_mean, df_bs_mean], axis=1)

        plt.figure(figsize=(4, 2))
        sns.heatmap(df_combined, annot=True, cmap="RdYlGn", linewidths=0.5)

        plt.title(f"Experimento {num_test} - Documentos {test_names[num_test-1]}", fontweight="bold", pad=10)
        plt.xlabel("Modelos", fontweight="bold")
        plt.ylabel("Métricas", fontweight="bold")
        plt.savefig(f"./test{num_test}/emb/experimento{num_test}_heatmap.png", bbox_inches="tight")

if __name__ == "__main__":
    main()