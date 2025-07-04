import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import mysql.connector
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def executar_apriori():
    try:
        # Conecta ao banco
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='ocorrencias_05'
        )

        query = "SELECT * FROM ocorrencias"
        df = pd.read_sql(query, conn)
        conn.close()

        # Prepara os dados
        df = df.drop(columns=['rótulos_de_linha', 'soma_da_linha'], errors='ignore')
        df = df.fillna(0).astype(int)

        # Executa Apriori
        freq_itemsets = apriori(df, min_support=0.1, use_colnames=True)
        regras = association_rules(freq_itemsets, metric="lift", min_threshold=1)

        # Limpa a exibição anterior
        output.delete("1.0", tk.END)

        # Mostra os itemsets frequentes
        output.insert(tk.END, "Itemsets Frequentes:\n")
        output.insert(tk.END, freq_itemsets.to_string(index=False))
        output.insert(tk.END, "\n\nRegras de Associação:\n")
        output.insert(tk.END, regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))

    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao executar Apriori:\n{e}")

# Criação da interface
janela = tk.Tk()
janela.title("Análise de Fatores Humanos - Apriori")
janela.geometry("1000x700")

# Botão
btn_executar = ttk.Button(janela, text="Executar Apriori", command=executar_apriori)
btn_executar.pack(pady=10)

# Área de saída
output = scrolledtext.ScrolledText(janela, wrap=tk.WORD, width=120, height=35)
output.pack(padx=10, pady=10)

janela.mainloop()
