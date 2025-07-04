import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Variáveis globais
modelo_global = None
feature_names = []

def executar_classificacao():
    global modelo_global, feature_names

    try:
        # Conectar ao banco
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='ocorrencias_05'
        )

        query = "SELECT * FROM ocorrencias"
        df = pd.read_sql(query, conn)
        conn.close()

        # Separar atributos e rótulo
        X = df.drop(columns=['rótulos_de_linha', 'soma_da_linha'], errors='ignore')
        y = df['soma_da_linha']
        feature_names = X.columns.tolist()

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Modelo: Árvore rasa com entropia
        modelo = DecisionTreeClassifier(random_state=42, max_depth=2, criterion='entropy')
        modelo.fit(X_train, y_train)
        modelo_global = modelo

        # Avaliação
        y_pred = modelo.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        matriz = confusion_matrix(y_test, y_pred)
        relatorio = classification_report(y_test, y_pred)

        # Exibir resultados na interface
        output.delete("1.0", tk.END)
        output.insert(tk.END, f"Acurácia: {acuracia:.2f}\n\n")
        output.insert(tk.END, "Matriz de Confusão:\n")
        output.insert(tk.END, f"{matriz}\n\n")
        output.insert(tk.END, "Relatório de Classificação:\n")
        output.insert(tk.END, relatorio)

    except Exception as e:
        messagebox.showerror("Erro", f"Erro durante a classificação:\n{e}")

def mostrar_arvore():
    global modelo_global, feature_names

    if modelo_global is None:
        messagebox.showwarning("Aviso", "Execute a classificação primeiro.")
        return

    try:
        plt.figure(figsize=(14, 8))  # Ajustado para árvore rasa
        plot_tree(modelo_global, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
        plt.title("Árvore de Decisão (max_depth=2, simplificada)", fontsize=14)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Erro ao gerar gráfico", str(e))

# Interface
janela = tk.Tk()
janela.title("Árvore de Decisão - Visualização Simplificada")
janela.geometry("900x650")

btn_classificar = ttk.Button(janela, text="Executar Classificação", command=executar_classificacao)
btn_classificar.pack(pady=5)

btn_mostrar = ttk.Button(janela, text="Visualizar Árvore", command=mostrar_arvore)
btn_mostrar.pack(pady=5)

output = scrolledtext.ScrolledText(janela, wrap=tk.WORD, width=100, height=35)
output.pack(padx=10, pady=10)

janela.mainloop()
