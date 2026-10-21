"""
Classe AvaliadorDesempenho (UC07 - Avaliar desempenho do modelo).

Calcula métricas de desempenho preditivo (acurácia, precisão, recall,
F1-score, AUC-ROC, AUC precisão-recall, erro de Brier e curva de
calibração), com intervalos de confiança de 95% por bootstrap
percentílico (reamostragem com reposição do conjunto de teste), e a
comparação pareada de AUC-ROC entre dois modelos avaliados no mesmo
conjunto de teste (mesmas reamostragens para os dois modelos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICAS_COM_IC = ("acuracia", "auc_roc", "auc_pr", "brier", "f1")


def _percentis(valores: list[float]) -> dict:
    if not valores:
        return {"ic_inferior": float("nan"), "ic_superior": float("nan")}
    return {
        "ic_inferior": float(np.percentile(valores, 2.5)),
        "ic_superior": float(np.percentile(valores, 97.5)),
    }


class AvaliadorDesempenho:
    def __init__(self, n_repeticoes_bootstrap: int = 1000, semente_aleatoria: int = 42):
        self.n_repeticoes_bootstrap = n_repeticoes_bootstrap
        self.semente_aleatoria = semente_aleatoria
        self.metricas: dict = {}

    @staticmethod
    def _validar_teste(y_real) -> None:
        """UC07-E01: conjunto de teste vazio ou com uma só classe não permite a avaliação."""
        y_real = np.asarray(y_real)
        if y_real.size == 0:
            raise ValueError("Conjunto de teste vazio: refaça a divisão dos dados (UC07-E01).")
        if len(np.unique(y_real)) < 2:
            raise ValueError("Conjunto de teste com uma só classe: a AUC-ROC não é definida; "
                             "refaça a divisão dos dados (UC07-E01).")

    def calcular_metricas(self, y_real, y_previsto, y_proba) -> dict:
        self._validar_teste(y_real)
        y_real = np.asarray(y_real)
        self.metricas = {
            "acuracia": float(accuracy_score(y_real, y_previsto)),
            "precisao": float(precision_score(y_real, y_previsto, zero_division=0)),
            "recall": float(recall_score(y_real, y_previsto, zero_division=0)),
            "f1": float(f1_score(y_real, y_previsto, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_real, y_proba)) if len(set(y_real)) > 1 else float("nan"),
            "auc_pr": float(average_precision_score(y_real, y_proba)),
            "brier": float(brier_score_loss(y_real, y_proba)),
        }
        return self.metricas

    def calcular_calibracao(self, y_real, y_proba, n_bins: int = 10) -> dict:
        frac_pos, media_prevista = calibration_curve(y_real, y_proba, n_bins=n_bins, strategy="quantile")
        return {"fracao_observada": frac_pos.tolist(), "probabilidade_media_prevista": media_prevista.tolist()}

    def calcular_intervalo_confianca_bootstrap(self, y_real, y_previsto, y_proba) -> dict:
        rng = np.random.default_rng(self.semente_aleatoria)
        y_real = np.asarray(y_real)
        y_previsto = np.asarray(y_previsto)
        y_proba = np.asarray(y_proba)
        n = len(y_real)
        amostras = {m: [] for m in METRICAS_COM_IC}
        for _ in range(self.n_repeticoes_bootstrap):
            idx = rng.integers(0, n, n)
            y_r, y_p, y_pb = y_real[idx], y_previsto[idx], y_proba[idx]
            if y_r.min() == y_r.max():
                continue
            amostras["acuracia"].append(accuracy_score(y_r, y_p))
            amostras["auc_roc"].append(roc_auc_score(y_r, y_pb))
            amostras["auc_pr"].append(average_precision_score(y_r, y_pb))
            amostras["brier"].append(brier_score_loss(y_r, y_pb))
            amostras["f1"].append(f1_score(y_r, y_p, zero_division=0))
        return {m: _percentis(v) for m, v in amostras.items()}

    def comparar_auc_pareado(self, y_real, y_proba_a, y_proba_b) -> dict:
        """
        Diferença de AUC-ROC (A − B) com IC de 95% por bootstrap pareado:
        em cada reamostragem, os DOIS modelos são avaliados nas mesmas
        observações, de modo que a variabilidade do conjunto de teste
        afeta os dois igualmente. Se o IC não contém zero, a diferença
        não é atribuível só à variação amostral do conjunto de teste.
        """
        rng = np.random.default_rng(self.semente_aleatoria)
        y_real = np.asarray(y_real)
        a = np.asarray(y_proba_a)
        b = np.asarray(y_proba_b)
        n = len(y_real)
        diferencas = []
        for _ in range(self.n_repeticoes_bootstrap):
            idx = rng.integers(0, n, n)
            if y_real[idx].min() == y_real[idx].max():
                continue
            diferencas.append(roc_auc_score(y_real[idx], a[idx]) - roc_auc_score(y_real[idx], b[idx]))
        return {
            "diferenca_auc": float(roc_auc_score(y_real, a) - roc_auc_score(y_real, b)),
            **_percentis(diferencas),
        }


def metricas_com_ic_em_colunas(metricas: dict, ic: dict) -> dict:
    """Achata {metrica: valor} + {metrica: {ic_inferior, ic_superior}} em colunas de CSV."""
    linha = dict(metricas)
    for m, limites in ic.items():
        linha[f"{m}_ic_inferior"] = limites["ic_inferior"]
        linha[f"{m}_ic_superior"] = limites["ic_superior"]
    return linha


def resumir_repeticoes(df: pd.DataFrame, chave: str, metricas: list[str]) -> pd.DataFrame:
    """Média, desvio-padrão, mínimo e máximo de cada métrica ao longo das repetições."""
    linhas = []
    for valor_chave, grupo in df.groupby(chave, sort=False):
        linha = {chave: valor_chave, "n_repeticoes": int(len(grupo))}
        for m in metricas:
            if m in grupo:
                serie = pd.to_numeric(grupo[m], errors="coerce")
                linha[f"{m}_media"] = float(serie.mean())
                linha[f"{m}_dp"] = float(serie.std(ddof=1)) if len(serie) > 1 else float("nan")
                linha[f"{m}_min"] = float(serie.min())
                linha[f"{m}_max"] = float(serie.max())
        linhas.append(linha)
    return pd.DataFrame(linhas)
