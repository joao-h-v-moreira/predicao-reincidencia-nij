"""
Classe RemovedorDeVies (parte do UC03/UC04 - Remover variáveis sensíveis
e proxies).

Remove a variável racial do conjunto de atributos preditivos, retendo-a
separadamente apenas para avaliação de equidade (UC08) -- nunca como
entrada do modelo. Identifica variáveis proxy por meio de dois critérios
estatísticos:

- V de Cramér, para variáveis categóricas, com limiar >= 0,20
  (associação moderada ou superior, conforme a escala de REA; PARKER,
  1992);
- correlação ponto-bisserial, para variáveis numéricas, com limiar
  |r| >= 0,30 (efeito moderado ou superior, conforme COHEN, 1988).

A remoção do atributo racial isoladamente (fairness through unawareness)
não é suficiente para evitar discriminação indireta via variáveis proxy
(PEDRESCHI; RUGGIERI; TURINI, 2008) -- daí a necessidade desta segunda
etapa de identificação de proxies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr


class RemovedorDeVies:
    def __init__(
        self,
        limiar_cramer_v: float = 0.20,
        limiar_correlacao_numerica: float = 0.30,
    ):
        self.limiar_cramer_v = limiar_cramer_v
        self.limiar_correlacao_numerica = limiar_correlacao_numerica
        self._raca_binaria: pd.Series | None = None

    def remover_variavel_racial(
        self, df: pd.DataFrame, coluna_raca: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        raca_retida = df[coluna_raca].copy()
        df_sem_raca = df.drop(columns=[coluna_raca])
        return df_sem_raca, raca_retida

    def obter_raca_para_avaliacao(self) -> pd.Series:
        if self._raca_binaria is None:
            raise RuntimeError(
                "Chame identificar_proxies_numericas() antes (define a "
                "codificação binária da raça usada na avaliação)."
            )
        return self._raca_binaria

    @staticmethod
    def _v_cramer(x: pd.Series, y: pd.Series) -> float:
        tabela = pd.crosstab(x, y)
        if tabela.size == 0 or tabela.shape[0] < 2 or tabela.shape[1] < 2:
            return 0.0
        chi2, _, _, _ = chi2_contingency(tabela)
        n = tabela.sum().sum()
        r, k = tabela.shape
        denom = min(r - 1, k - 1)
        if denom == 0 or n == 0:
            return 0.0
        return float(np.sqrt((chi2 / n) / denom))

    def identificar_proxies_categoricas(
        self, df: pd.DataFrame, raca: pd.Series
    ) -> list[str]:
        proxies = []
        categoricas = df.select_dtypes(exclude="number").columns
        for col in categoricas:
            v = self._v_cramer(df[col].astype("object").fillna("NA"), raca)
            if v >= self.limiar_cramer_v:
                proxies.append(col)
        return proxies

    def identificar_proxies_numericas(
        self, df: pd.DataFrame, raca: pd.Series
    ) -> list[str]:
        # codifica a raça (binária Black/White, conforme o codebook do
        # NIJ -- limitação documentada no trabalho) para a correlação
        # ponto-bisserial
        categorias = pd.Categorical(raca)
        raca_binaria = pd.Series(categorias.codes, index=raca.index)
        self._raca_binaria = raca_binaria

        proxies = []
        numericas = df.select_dtypes("number").columns
        for col in numericas:
            valores = df[col]
            mascara = valores.notna() & (raca_binaria >= 0)
            if mascara.sum() < 3 or valores[mascara].nunique() < 2:
                continue
            r, _ = pointbiserialr(raca_binaria[mascara], valores[mascara])
            if abs(r) >= self.limiar_correlacao_numerica:
                proxies.append(col)
        return proxies

    @staticmethod
    def aplicar_remocao(df: pd.DataFrame, colunas_proxy: list[str]) -> pd.DataFrame:
        return df.drop(columns=colunas_proxy, errors="ignore")

    def diagnostico_associacao(self, df: pd.DataFrame, raca: pd.Series) -> pd.DataFrame:
        """
        Calcula a associação de CADA variável remanescente com a raça
        (V de Cramér para categóricas, ponto-bisserial para numéricas),
        independentemente de ultrapassar o limiar ou não. Ao contrário de
        identificar_proxies_categoricas/numericas (que só devolvem as
        que PASSAM do limiar), este método existe para auditoria: expõe
        o valor de associação de toda variável testada, para que seja
        possível verificar quão perto do limiar cada uma ficou -- o
        teste bivariado (variável a variável) não captura proxies
        combinatórios (vários atributos fracamente associados que,
        somados, reconstroem a raça), então essa tabela é o que permite
        inspecionar essa limitação, em vez de confiar cegamente no
        resultado binário passou/não passou.
        """
        categorias = pd.Categorical(raca)
        raca_binaria = pd.Series(categorias.codes, index=raca.index)

        linhas = []
        for col in df.select_dtypes(exclude="number").columns:
            v = self._v_cramer(df[col].astype("object").fillna("NA"), raca)
            linhas.append({
                "variavel": col, "tipo": "categorica", "teste": "V de Cramer",
                "valor": v, "limiar": self.limiar_cramer_v,
                "ultrapassa_limiar": v >= self.limiar_cramer_v,
            })
        for col in df.select_dtypes("number").columns:
            valores = df[col]
            mascara = valores.notna() & (raca_binaria >= 0)
            if mascara.sum() < 3 or valores[mascara].nunique() < 2:
                r = float("nan")
            else:
                r, _ = pointbiserialr(raca_binaria[mascara], valores[mascara])
            linhas.append({
                "variavel": col, "tipo": "numerica", "teste": "ponto-bisserial",
                "valor": abs(r) if r == r else r, "limiar": self.limiar_correlacao_numerica,
                "ultrapassa_limiar": (abs(r) >= self.limiar_correlacao_numerica) if r == r else False,
            })

        resultado = pd.DataFrame(linhas).sort_values("valor", ascending=False, na_position="last")
        return resultado.reset_index(drop=True)
