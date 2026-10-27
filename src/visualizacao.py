"""
Gráficos de avaliação: curvas ROC, curvas de calibração (por modelo) e
curvas de calibração por grupo protegido (para um modelo).

Mantido como módulo separado da avaliação numérica para isolar a
dependência de matplotlib. Todos os números nas figuras usam vírgula
decimal, como o texto do TCC.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve

CORES_POR_MODELO = {
    "regressao_logistica": "#1f77b4",
    "random_forest": "#2ca02c",
    "gradient_boosting": "#d62728",
    "baseline_dressel_farid": "#7f7f7f",
}

NOMES_EXIBICAO = {
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "baseline_dressel_farid": "Baseline (Dressel e Farid, 2018)",
}

CORES_GRUPOS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]


def _num(v: float, casas: int = 3) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _eixos_com_virgula(ax) -> None:
    formato = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)


def plotar_curvas_roc(y_teste, probabilidades_por_modelo: dict, caminho_saida="results/curvas_roc.png") -> Path:
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome_modelo, y_proba in probabilidades_por_modelo.items():
        fpr, tpr, _ = roc_curve(y_teste, y_proba)
        rotulo = NOMES_EXIBICAO.get(nome_modelo, nome_modelo)
        ax.plot(fpr, tpr, color=CORES_POR_MODELO.get(nome_modelo), linewidth=2,
                label=f"{rotulo} (AUC = {_num(auc(fpr, tpr))})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Classificador aleatório (AUC = 0,500)")
    ax.set_xlabel("Taxa de falsos positivos (FPR)")
    ax.set_ylabel("Taxa de verdadeiros positivos (TPR)")
    ax.set_title("Curvas ROC – comparação entre modelos")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    _eixos_com_virgula(ax)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def plotar_curvas_calibracao(y_teste, probabilidades_por_modelo: dict, caminho_saida="results/curvas_calibracao.png", n_bins: int = 10) -> Path:
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome_modelo, y_proba in probabilidades_por_modelo.items():
        frac, media = calibration_curve(y_teste, y_proba, n_bins=n_bins, strategy="quantile")
        ax.plot(media, frac, marker="o", color=CORES_POR_MODELO.get(nome_modelo), linewidth=2,
                label=NOMES_EXIBICAO.get(nome_modelo, nome_modelo))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Calibração perfeita")
    ax.set_xlabel("Probabilidade média prevista")
    ax.set_ylabel("Fração observada de reincidência")
    ax.set_title("Curvas de calibração – comparação entre modelos")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    _eixos_com_virgula(ax)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def plotar_calibracao_por_grupo(y_teste, y_proba, grupo, nome_modelo: str, caminho_saida="results/calibracao_por_grupo.png", n_bins: int = 10) -> Path:
    """Curva de calibração de UM modelo, separada por grupo protegido."""
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    y_teste = np.asarray(y_teste)
    y_proba = np.asarray(y_proba)
    g = pd.Series(grupo).astype(object).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    grupos = sorted({v for v in g if isinstance(v, str)})
    for i, v in enumerate(grupos):
        m = g == v
        frac, media = calibration_curve(y_teste[m], y_proba[m], n_bins=n_bins, strategy="quantile")
        ax.plot(media, frac, marker="o", linewidth=2, color=CORES_GRUPOS[i % len(CORES_GRUPOS)],
                label=f"{v} (n = {int(m.sum())})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Calibração perfeita")
    ax.set_xlabel("Probabilidade média prevista")
    ax.set_ylabel("Fração observada de reincidência")
    ax.set_title(f"Calibração por grupo racial – {NOMES_EXIBICAO.get(nome_modelo, nome_modelo)}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    _eixos_com_virgula(ax)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def plotar_intervalos(linhas: list[dict], titulo: str, rotulo_x: str, caminho_saida, referencia: float = 0.0,
                      escala: float = 100.0, casas: int = 1) -> Path:
    """
    Gráfico de intervalos (forest plot): cada linha tem "rotulo", "valor",
    "ic_inferior", "ic_superior" e, opcionalmente, "serie" (para
    diferenciar cores/marcadores, ex.: antes × depois). Os valores são
    multiplicados por `escala` (100 = pontos percentuais). A linha
    vertical tracejada marca o valor de igualdade (`referencia`).
    """
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    rotulos = list(dict.fromkeys(l["rotulo"] for l in linhas))
    series = list(dict.fromkeys(l.get("serie", "") for l in linhas))
    # primeira série desenhada acima das demais em cada linha
    deslocamento = {s: ((len(series) - 1) / 2 - i) * 0.22 for i, s in enumerate(series)}
    marcadores = ["o", "s", "D", "^"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rotulos) + 1.6))
    for i_s, s in enumerate(series):
        ys, xs, erros = [], [], [[], []]
        for l in linhas:
            if l.get("serie", "") != s:
                continue
            y = len(rotulos) - 1 - rotulos.index(l["rotulo"]) + deslocamento[s]
            v, lo, hi = (escala * l[k] for k in ("valor", "ic_inferior", "ic_superior"))
            ys.append(y)
            xs.append(v)
            erros[0].append(v - lo)
            erros[1].append(hi - v)
        ax.errorbar(xs, ys, xerr=erros, fmt=marcadores[i_s % len(marcadores)], color=CORES_GRUPOS[i_s % len(CORES_GRUPOS)],
                    capsize=3, linewidth=1.5, label=s or None)
    ax.axvline(escala * referencia, linestyle="--", color="gray", linewidth=1)
    ax.set_yticks(range(len(rotulos)))
    ax.set_yticklabels(list(reversed(rotulos)), fontsize=9)
    ax.set_xlabel(rotulo_x)
    fig.suptitle(titulo, fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{casas}f}".replace(".", ",")))
    if any(series):
        ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida
