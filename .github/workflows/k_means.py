import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import mysql.connector
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Variáveis globais para reuso nos dois botões
df_clusters = None
kmeans_model = None

def executar_kmeans():
    global df_clusters, kmeans_model
    try:
        # Conecta ao banco
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='ocorrencias_05'
        )

        # Carrega os dados
        query = "SELECT * FROM ocorrencias"
        df = pd.read_sql(query, conn)
        conn.close()

        # Remove colunas não numéricas ou irrelevantes
        X = df.drop(columns=['rótulos_de_linha', 'soma_da_linha'], errors='ignore')
        X = X.fillna(0)

        # Padroniza
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Aplica K-means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df['cluster'] = clusters
        df_clusters = df.copy()
        kmeans_model = kmeans

        # Exibe resultados
        contagem = df['cluster'].value_counts().sort_index()
        centroides = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

        output.delete("1.0", tk.END)
        output.insert(tk.END, "Contagem de registros por cluster:\n")
        for idx, count in contagem.items():
            output.insert(tk.END, f"Cluster {idx}: {count} registros\n")

        output.insert(tk.END, "\nCentroides (valores padronizados):\n")
        output.insert(tk.END, centroides.to_string(index=True))

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao aplicar K-means:\n{e}")

def mostrar_grafico_2d():
    global df_clusters, kmeans_model
    if df_clusters is None or kmeans_model is None:
        messagebox.showwarning("Aviso", "Execute o K-means primeiro.")
        return

    try:
        X = df_clusters.drop(columns=['rótulos_de_linha', 'soma_da_linha', 'cluster'], errors='ignore')
        X_scaled = StandardScaler().fit_transform(X)

        # Reduz para 2D com PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        for cluster_id in sorted(df_clusters['cluster'].unique()):
            plt.scatter(
                X_pca[df_clusters['cluster'] == cluster_id, 0],
                X_pca[df_clusters['cluster'] == cluster_id, 1],
                label=f'Cluster {cluster_id}'
            )

        plt.title("Clusters K-means (reduzidos para 2D com PCA)")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Erro ao gerar gráfico", str(e))

# Interface Tkinter
janela = tk.Tk()
janela.title("Clusterização - K-means com Visualização 2D")
janela.geometry("1000x700")

btn_kmeans = ttk.Button(janela, text="Executar K-means", command=executar_kmeans)
btn_kmeans.pack(pady=5)

btn_visualizar = ttk.Button(janela, text="Visualizar Gráfico 2D", command=mostrar_grafico_2d)
btn_visualizar.pack(pady=5)

output = scrolledtext.ScrolledText(janela, wrap=tk.WORD, width=120, height=35)
output.pack(padx=10, pady=10)

janela.mainloop()
