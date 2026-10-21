"""
Classe ComparadorModelos (UC09 - Comparar modelos).

Consolida as avaliações de vários modelos numa tabela única. Só compara
modelos avaliados no MESMO conjunto de teste (UC09-E01): cada avaliação
traz a identificação do conjunto de teste (semente da divisão e tamanho);
se elas divergirem, a comparação é interrompida.
"""
from __future__ import annotations

import pandas as pd


class ComparadorModelos:
    def comparar(self, avaliacoes: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(avaliacoes)
        if {"semente_divisao", "n_teste"} <= set(df.columns):
            conjuntos = df[["semente_divisao", "n_teste"]].drop_duplicates()
            if len(conjuntos) > 1:
                raise ValueError(
                    "Modelos avaliados em conjuntos de teste diferentes não podem ser "
                    f"comparados diretamente: {conjuntos.to_dict(orient='records')}"
                )
        colunas_ordenadas = [c for c in ["modelo", "acuracia", "precisao", "recall", "f1", "auc_roc", "auc_pr", "brier"] if c in df.columns]
        outras = [c for c in df.columns if c not in colunas_ordenadas]
        return df[colunas_ordenadas + outras]
