"""
Classe DivisorDados (UC05 - Selecionar e configurar algoritmos, parte da
divisão de dados).

Realiza a divisão treino/teste estratificada e a validação cruzada
aninhada (a busca de hiperparâmetros ocorre nas dobras internas de
validação, nunca no conjunto de teste, reservado para a avaliação final
em UC07/UC08).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class DivisorDados:
    def __init__(self, proporcao_teste: float = 0.3, semente_aleatoria: int = 42):
        self.proporcao_teste = proporcao_teste
        self.semente_aleatoria = semente_aleatoria

    def dividir_treino_teste(
        self, x: pd.DataFrame, y: pd.Series, estratificar_por: pd.Series | None = None,
        semente: int | None = None,
    ):
        estrato = estratificar_por if estratificar_por is not None else y
        return train_test_split(
            x,
            y,
            test_size=self.proporcao_teste,
            random_state=self.semente_aleatoria if semente is None else semente,
            stratify=estrato,
        )

    def validacao_cruzada_aninhada(self, n_repeticoes: int = 10, k_interno: int = 3):
        """
        Validação aninhada usada no protocolo (Seção 2.5):

        - laço EXTERNO: `n_repeticoes` divisões treino/teste aleatórias e
          estratificadas (70/30), com sementes semente_aleatoria,
          semente_aleatoria + 1, ... (validação cruzada de Monte Carlo).
          A primeira divisão (semente 42) é a "divisão de referência",
          usada nas figuras e nas análises detalhadas;
        - laço INTERNO: k-fold estratificado (k = 3), aplicado só dentro
          do treino de cada divisão externa, para a busca de
          hiperparâmetros (ConfiguradorAlgoritmo.buscar_hiperparametros).

        Retorna (lista de sementes do laço externo, splitter interno).
        """
        sementes = [self.semente_aleatoria + i for i in range(n_repeticoes)]
        interno = StratifiedKFold(n_splits=k_interno, shuffle=True, random_state=self.semente_aleatoria)
        return sementes, interno
