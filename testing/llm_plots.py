import pandas as pd
import matplotlib.pyplot as plt

def upload_data(model, cases):
    dfs = []
    for i, case in enumerate(cases, start=1):
        if i == 3:
            filename = f"./test{i}/llm/test{i}_{model}_multilingual_large.csv"
        else:
            filename = f"./test{i}/llm/test{i}_{model}_mxbai_large.csv"

        df = pd.read_csv(filename)
        df.fillna(0, inplace=True)
        df = df.drop(columns=["question", "answer", "contexts", "ground_truth", "context_precision", "context_recall"])
        df_mean = df.mean().to_frame(f"CE {i}: {case}")
        dfs.append(df_mean)
    return pd.concat(dfs, axis=1)

def graphics(df_gemma, df_llama, df_mistral, df_phi, metric, title, file_name, legend_loc):
    plt.figure(figsize=(6, 5))

    df_gemma = df_gemma.T
    df_llama = df_llama.T
    df_mistral = df_mistral.T
    df_phi = df_phi.T

    plt.plot(df_gemma.index, df_gemma[metric], marker='o', label="Gemma")
    plt.plot(df_llama.index, df_llama[metric], marker='o', label="Llama 3")
    plt.plot(df_mistral.index, df_mistral[metric], marker='o', label="Mistral")
    plt.plot(df_phi.index, df_phi[metric], marker='o', label="Phi 3")

    plt.title(title, loc="center", pad=15, fontweight="bold")
    if metric == "faithfulness":
        plt.ylabel("Faithfulness")
        plt.legend(loc=legend_loc)
    else:
        plt.ylabel("Answer Relevancy")
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=legend_loc)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(file_name)

def main():
    cases = ["Documentos \njurídicos", "Documentos \nfinancieros", "Documentos \ncientífico-técnicos"]

    df_gemma = upload_data("gemma", cases)
    df_llama = upload_data("llama3", cases)
    df_mistral = upload_data("mistral", cases)
    df_phi = upload_data("phi3", cases)

    graphics(df_gemma, df_llama, df_mistral, df_phi, "faithfulness",
                     "Faithfulness de los LLM por caso de estudio (CE)", "llm_faithfulness.png", 'upper center')

    graphics(df_gemma, df_llama, df_mistral, df_phi, "answer_relevancy",
                     "Answer Relevancy de los LLM por caso de estudio (CE)", "llm_answer_relevancy.png", 'center right')

if __name__ == "__main__":
    main()
