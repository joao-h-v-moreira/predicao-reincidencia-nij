"""
Classe AnalisadorExploratorio (UC02 - Realizar análise exploratória / EDA).

"EDA" aqui significa Análise Exploratória de Dados (Exploratory Data
Analysis) -- não confundir com a arquitetura orientada a eventos
(Event-Driven Architecture), também referenciada neste projeto.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Colunas que, mesmo sendo numéricas, não têm valor analítico para uma
# EDA -- ID é só um identificador de linha (um número sequencial sem
# significado substantivo), então um histograma da sua distribuição não
# diz nada sobre o fenômeno estudado.
COLUNAS_NAO_INFORMATIVAS_PARA_EDA = ["ID"]

# Quando houver mais de uma coluna numérica candidata ao segundo
# histograma, prioriza-se uma com significado substantivo para o tema do
# trabalho (score de risco calculado pela supervisão), em vez de pegar
# "a primeira que aparecer no arquivo" sem critério.
COLUNAS_PREFERIDAS_PARA_EDA = [
    "Supervision_Risk_Score_First",
    "Percent_Days_Employed",
    "Prison_Years",
]


class AnalisadorExploratorio:
    def __init__(self, dataframe: pd.DataFrame, coluna_alvo: str, colunas: list[str] | None = None):
        """
        `colunas` (opcional, UC02-FA01): restringe a análise a um
        subconjunto de colunas de interesse (a coluna-alvo é sempre
        incluída). Colunas inexistentes geram erro (UC02-E01).
        """
        if colunas is not None:
            inexistentes = [c for c in colunas if c not in dataframe.columns]
            if inexistentes:
                raise ValueError(f"Coluna(s) inexistente(s) informada(s) para a EDA: {inexistentes}")
            dataframe = dataframe[list(dict.fromkeys([*colunas, coluna_alvo]))]
        self.dataframe = dataframe
        self.coluna_alvo = coluna_alvo

    def gerar_estatisticas_descritivas(self) -> dict:
        return {
            "numericas": self.dataframe.select_dtypes("number").describe().to_dict(),
            "categoricas": {
                col: self.dataframe[col].value_counts(dropna=False).to_dict()
                for col in self.dataframe.select_dtypes(exclude="number").columns
            },
        }

    def calcular_valores_ausentes(self) -> dict:
        ausentes = self.dataframe.isna().mean().sort_values(ascending=False)
        return ausentes[ausentes > 0].to_dict()

    def plotar_distribuicoes(self, caminho_saida: str | Path = "results/eda_distribuicoes.png") -> Path:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        contagem_alvo = self.dataframe[self.coluna_alvo].value_counts().sort_index()
        contagem_alvo.plot(kind="bar", ax=axes[0], color="#33475B", rot=0)
        axes[0].set_title("Distribuição da variável-alvo")
        axes[0].set_xlabel(self.coluna_alvo)
        axes[0].set_ylabel("Frequência")

        numericas = self.dataframe.select_dtypes("number").drop(
            columns=[self.coluna_alvo, *COLUNAS_NAO_INFORMATIVAS_PARA_EDA], errors="ignore"
        )
        if not numericas.empty:
            coluna_escolhida = next(
                (c for c in COLUNAS_PREFERIDAS_PARA_EDA if c in numericas.columns),
                numericas.columns[0],
            )
            serie = numericas[coluna_escolhida].dropna()
            # variável de valores inteiros (ex.: score de 1 a 10): uma barra
            # por valor; um histograma com faixas arbitrárias criaria faixas
            # vazias artificiais entre valores inteiros
            if len(serie) and (serie == serie.round()).all() and serie.nunique() <= 30:
                serie.astype(int).value_counts().sort_index().plot(kind="bar", ax=axes[1], color="#33475B", rot=0)
            else:
                serie.plot(kind="hist", bins=20, ax=axes[1], color="#33475B")
            axes[1].set_title(f"Distribuição de {coluna_escolhida}")
            axes[1].set_xlabel(coluna_escolhida)
            axes[1].set_ylabel("Frequência")

        fig.tight_layout()
        fig.savefig(caminho_saida, dpi=120)
        plt.close(fig)
        return caminho_saida
