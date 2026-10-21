"""
Classe AvaliadorAblacao (UC09 - Comparar modelos, estudo de ablação).

Compara quatro variantes do conjunto de atributos preditivos:
  (i)   com variável racial e com variáveis proxy;
  (ii)  com variável racial e sem variáveis proxy;
  (iii) sem variável racial e com variáveis proxy;
  (iv)  sem variável racial e sem variáveis proxy (variante principal
        do projeto).

Quantifica o efeito de cada decisão de remoção sobre o desempenho
preditivo e sobre as métricas de equidade.
"""
from __future__ import annotations

import pandas as pd

VARIANTES = [
    "com_raca_com_proxies",
    "com_raca_sem_proxies",
    "sem_raca_com_proxies",
    "sem_raca_sem_proxies",
]


class AvaliadorAblacao:
    def __init__(self, variantes: list[str] | None = None):
        self.variantes = variantes or VARIANTES
        self.resultados: dict[str, dict] = {}

    def executar_ablacao(self, variante: str, resultado_desempenho: dict, resultado_equidade: dict) -> dict:
        registro = {"variante": variante, **resultado_desempenho, **resultado_equidade}
        self.resultados[variante] = registro
        return registro

    def comparar_variantes(self) -> pd.DataFrame:
        if not self.resultados:
            return pd.DataFrame()
        return pd.DataFrame(self.resultados.values())
