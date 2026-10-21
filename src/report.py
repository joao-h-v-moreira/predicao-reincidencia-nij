"""
Classe RelatorioResultados (UC10 - Gerar e exportar relatório de
resultados).

Reúne as tabelas produzidas pelo pipeline (comparação de modelos,
equidade detalhada, ablação, repetições, sensibilidade etc.) e exporta
cada uma como CSV na pasta de resultados.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class RelatorioResultados:
    def __init__(self, formato: str = "csv"):
        self.formato = formato

    def gerar(self, comparacao: pd.DataFrame, ablacao: pd.DataFrame, **outras_tabelas: pd.DataFrame) -> dict:
        return {"comparacao_modelos": comparacao, "estudo_ablacao": ablacao, **outras_tabelas}

    def exportar(self, resultados: dict, caminho: str | Path = "results") -> list[Path]:
        caminho = Path(caminho)
        caminho.mkdir(parents=True, exist_ok=True)
        arquivos = []
        for nome, df in resultados.items():
            destino = caminho / f"{nome}.csv"
            df.to_csv(destino, index=False)
            arquivos.append(destino)
        return arquivos
