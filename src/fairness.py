"""
Classe AvaliadorEquidade (UC08 - Avaliar métricas de equidade).

Métricas por grupo protegido (raça e sexo): taxa-base, taxa de predição
de alto risco, FPR, FNR, TPR, PPV, calibração por grupo, e as diferenças
entre grupos, com intervalos de confiança de 95% por bootstrap.

Convenções (registradas no texto do TCC, Seção 2.5):
- "Positivo" = o modelo prevê reincidência (alto risco). Esse é o
  desfecho DESFAVORÁVEL para a pessoa avaliada.
- Todas as comparações entre grupos são DIRECIONAIS, numa ordem fixa
  (ORDEM_GRUPOS): grupo A = BLACK, grupo B = WHITE (raça); grupo A = M,
  grupo B = F (sexo). O grupo B é o de referência.
- disparate_impact_ratio = taxa de predição FAVORÁVEL (baixo risco) do
  grupo A / a do grupo B. É a convenção da regra dos quatro quintos,
  formulada para desfechos favoráveis: valores abaixo de 0,80 indicam
  que o grupo A recebe a classificação favorável com frequência menor
  que 80% da do grupo B; valores acima de 1 indicam a direção inversa
  (e acima de 1,25, alerta na direção inversa).
- diferenca_paridade_demografica = taxa de alto risco de A − a de B.
- dif_fpr, dif_fnr, dif_tpr, dif_ppv = valor do grupo A − valor do
  grupo B (diferença COM SINAL).

Por que diferenças com sinal (e não "maior − menor"): uma medida do tipo
|A − B| nunca é negativa, de modo que sua distribuição bootstrap fica
"dobrada" sobre o zero e o intervalo de confiança percentílico nunca
contém zero, mesmo quando não há disparidade alguma (o mesmo ocorre com
a razão mín/máx, que nunca passa de 1). Com a diferença com sinal, o IC
pode conter zero (ou 1, para a razão), e a pergunta "a disparidade é
distinguível da variação amostral?" passa a ter resposta válida; o sinal
mostra, além disso, qual grupo é desfavorecido.

Os critérios de equidade não podem, em geral, ser satisfeitos ao mesmo
tempo quando as taxas-base diferem entre grupos (CHOULDECHOVA, 2017;
KLEINBERG; MULLAINATHAN; RAGHAVAN, 2017); por isso o pipeline reporta
várias métricas complementares, e decompõe a diferença de FPR entre
grupos (decompor_disparidade_fpr) segundo a identidade de Chouldechova.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

N_MINIMO_SUBGRUPO = 30

# Ordem fixa das comparações entre grupos: (grupo A, grupo B de referência)
ORDEM_GRUPOS = {"raca": ("BLACK", "WHITE"), "sexo": ("M", "F")}
METRICAS_ENTRE_GRUPOS = ("disparate_impact_ratio", "diferenca_paridade_demografica",
                         "dif_fpr", "dif_fnr", "dif_tpr", "dif_ppv")


def _metricas_grupo(yr: np.ndarray, yp: np.ndarray) -> dict:
    n = len(yr)
    positivos = int((yr == 1).sum())
    negativos = n - positivos
    vp = int(((yp == 1) & (yr == 1)).sum())
    fp = int(((yp == 1) & (yr == 0)).sum())
    fn = positivos - vp
    preditos_pos = vp + fp
    return {
        "n": n,
        "taxa_base": positivos / n if n else float("nan"),
        "taxa_alto_risco": preditos_pos / n if n else float("nan"),
        "taxa_baixo_risco": 1 - preditos_pos / n if n else float("nan"),
        "fpr": fp / negativos if negativos else float("nan"),
        "fnr": fn / positivos if positivos else float("nan"),
        "tpr": vp / positivos if positivos else float("nan"),
        "ppv": vp / preditos_pos if preditos_pos else float("nan"),
    }


def _resumo_entre_grupos(por_grupo: dict, ordem: tuple[str, str]) -> dict:
    """Comparações direcionais grupo A vs grupo B (ordem fixa)."""
    a, b = ordem
    if a not in por_grupo or b not in por_grupo:
        return {m: float("nan") for m in METRICAS_ENTRE_GRUPOS}
    ma, mb = por_grupo[a], por_grupo[b]
    resultado = {
        "disparate_impact_ratio": (ma["taxa_baixo_risco"] / mb["taxa_baixo_risco"]
                                   if mb["taxa_baixo_risco"] else float("nan")),
        "diferenca_paridade_demografica": ma["taxa_alto_risco"] - mb["taxa_alto_risco"],
    }
    for m in ("fpr", "fnr", "tpr", "ppv"):
        resultado[f"dif_{m}"] = ma[m] - mb[m]
    return resultado


def _ece(yr: np.ndarray, yp: np.ndarray, n_bins: int) -> float:
    """Erro de calibração esperado, com faixas por quantis das probabilidades."""
    if len(yr) < n_bins:
        return float("nan")
    cortes = np.unique(np.quantile(yp, np.linspace(0, 1, n_bins + 1)))
    faixas = np.clip(np.searchsorted(cortes, yp, side="right") - 1, 0, len(cortes) - 2)
    total = 0.0
    for b in np.unique(faixas):
        m = faixas == b
        total += m.sum() / len(yr) * abs(yp[m].mean() - yr[m].mean())
    return float(total)


class AvaliadorEquidade:
    def __init__(self, atributos_protegidos: list[str], n_repeticoes_bootstrap: int = 1000, semente_aleatoria: int = 42):
        self.atributos_protegidos = atributos_protegidos
        self.n_repeticoes_bootstrap = n_repeticoes_bootstrap
        self.semente_aleatoria = semente_aleatoria
        self.metricas_equidade: dict = {}

    @staticmethod
    def _arrays(y_real, y_previsto, grupo):
        g = pd.Series(grupo).astype(object).to_numpy()
        return np.asarray(y_real).astype(int), np.asarray(y_previsto).astype(int), g

    @staticmethod
    def _grupos_validos(g: np.ndarray) -> list:
        return sorted({v for v in g if isinstance(v, str) or not (isinstance(v, float) and math.isnan(v))}, key=str)

    def _ordem(self, g: np.ndarray, ordem) -> tuple[str, str]:
        """Ordem (A, B) informada; sem ela, só é definida se houver exatamente dois grupos."""
        if ordem is not None:
            return tuple(ordem)
        grupos = self._grupos_validos(g)
        if len(grupos) != 2:
            raise ValueError(f"Informe a ordem (grupo A, grupo B) para comparar os grupos {grupos}.")
        return grupos[0], grupos[1]

    def calcular_equidade(self, y_real, y_previsto, grupo, ordem: tuple[str, str] | None = None) -> dict:
        yr, yp, g = self._arrays(y_real, y_previsto, grupo)
        ordem = self._ordem(g, ordem)
        por_grupo = {str(v): _metricas_grupo(yr[g == v], yp[g == v]) for v in self._grupos_validos(g)}
        pequenos = {k: m["n"] for k, m in por_grupo.items() if m["n"] < N_MINIMO_SUBGRUPO}
        if pequenos:
            # UC08-E01: métricas de subgrupos pequenos têm baixa confiabilidade
            warnings.warn(f"Subgrupo(s) com menos de {N_MINIMO_SUBGRUPO} registros: {pequenos}; "
                          "interprete as métricas desses grupos com cautela (ver ICs).")
        resultado = {"por_grupo": por_grupo, "grupo_a": ordem[0], "grupo_b": ordem[1],
                     **_resumo_entre_grupos(por_grupo, ordem)}
        # chaves mantidas por compatibilidade com o restante do pipeline
        resultado["fpr_por_grupo"] = {k: v["fpr"] for k, v in por_grupo.items()}
        resultado["tpr_por_grupo"] = {k: v["tpr"] for k, v in por_grupo.items()}
        resultado["taxa_predicao_positiva_por_grupo"] = {k: v["taxa_alto_risco"] for k, v in por_grupo.items()}
        self.metricas_equidade = resultado
        return resultado

    def calcular_taxas_erro_por_grupo(self, y_real, y_previsto, grupo, ordem=None) -> dict:
        por_grupo = self.calcular_equidade(y_real, y_previsto, grupo, ordem)["por_grupo"]
        return {k: {"fpr": v["fpr"], "fnr": v["fnr"], "n": v["n"]} for k, v in por_grupo.items()}

    def calcular_calibracao_por_grupo(self, y_real, y_proba, grupo, n_bins: int = 10) -> dict:
        """
        Calibração DENTRO de cada grupo. O erro de Brier sozinho não mede
        calibração (mistura calibração e capacidade de discriminação, e
        depende da taxa-base do grupo), então também são reportados:
        - calibracao_no_agregado: probabilidade média prevista − taxa
          observada no grupo (positivo = o modelo superestima o risco do
          grupo);
        - ece: erro de calibração esperado em 10 faixas de quantis;
        - brier_skill: 1 − Brier / [p(1−p)], isto é, o Brier relativo ao
          de um modelo que previsse sempre a taxa-base do grupo.
        """
        yr = np.asarray(y_real).astype(int)
        yp = np.asarray(y_proba).astype(float)
        g = pd.Series(grupo).astype(object).to_numpy()
        resultado = {}
        for v in self._grupos_validos(g):
            m = g == v
            r, p = yr[m], yp[m]
            taxa = r.mean()
            brier = float(brier_score_loss(r, p)) if len(np.unique(r)) > 1 else float("nan")
            referencia = taxa * (1 - taxa)
            resultado[str(v)] = {
                "n": int(m.sum()),
                "taxa_observada": float(taxa),
                "media_prevista": float(p.mean()),
                "calibracao_no_agregado": float(p.mean() - taxa),
                "ece": _ece(r, p, n_bins),
                "brier": brier,
                "brier_referencia": float(referencia),
                "brier_skill": float(1 - brier / referencia) if referencia > 0 else float("nan"),
            }
        return resultado

    def calcular_intervalo_confianca_bootstrap(self, y_real, y_previsto, grupo, y_proba=None,
                                               ordem: tuple[str, str] | None = None) -> dict:
        """
        IC de 95% (bootstrap percentílico, reamostrando o conjunto de
        teste inteiro) para as métricas por grupo e para as comparações
        direcionais entre grupos (A − B e razão A/B). Se o IC de uma
        diferença dif_* não contém zero (ou o do disparate_impact_ratio
        não contém 1), a disparidade não é atribuível só à variação
        amostral do conjunto de teste.
        """
        yr, yp, g = self._arrays(y_real, y_previsto, grupo)
        ordem = self._ordem(g, ordem)
        proba = None if y_proba is None else np.asarray(y_proba).astype(float)
        grupos = self._grupos_validos(g)
        rng = np.random.default_rng(self.semente_aleatoria)
        n = len(yr)
        amostras: dict[str, list] = {}

        def guardar(chave, valor):
            if valor is not None and not (isinstance(valor, float) and math.isnan(valor)):
                amostras.setdefault(chave, []).append(valor)

        for _ in range(self.n_repeticoes_bootstrap):
            idx = rng.integers(0, n, n)
            r, p, gg = yr[idx], yp[idx], g[idx]
            por_grupo = {}
            for v in grupos:
                m = gg == v
                if m.sum() == 0:
                    continue
                por_grupo[v] = _metricas_grupo(r[m], p[m])
                for met in ("fpr", "fnr", "tpr", "ppv", "taxa_alto_risco"):
                    guardar(f"{v}__{met}", por_grupo[v][met])
                if proba is not None and len(np.unique(r[m])) > 1:
                    guardar(f"{v}__brier", float(brier_score_loss(r[m], proba[idx][m])))
                    guardar(f"{v}__calibracao_no_agregado", float(proba[idx][m].mean() - r[m].mean()))
            if len(por_grupo) >= 2:
                for chave, valor in _resumo_entre_grupos(por_grupo, ordem).items():
                    guardar(chave, valor)

        return {
            chave: {"ic_inferior": float(np.percentile(v, 2.5)), "ic_superior": float(np.percentile(v, 97.5))}
            for chave, v in amostras.items()
        }

    def decompor_disparidade_fpr(self, y_real, y_previsto, grupo, ordem: tuple[str, str] | None = None) -> dict:
        """
        Decomposição da razão de FPR entre dois grupos pela identidade de
        Chouldechova (2017), que vale exatamente para qualquer
        classificador:

            FPR = [p / (1 − p)] × [(1 − PPV) / PPV] × (1 − FNR)

        em que p é a taxa-base do grupo. Logo, a razão FPR_a / FPR_b é o
        produto de três razões: (i) das chances da taxa-base, (ii) do
        termo de precisão (PPV) e (iii) do termo de sensibilidade (1 −
        FNR). Tomando o logaritmo, cada fator responde por uma parcela
        aditiva da diferença, o que permite dizer QUANTO da disparidade
        de FPR vem da diferença de taxa-base (que o modelo não controla)
        e quanto vem de o modelo ser menos preciso ou mais sensível para
        um dos grupos. Os grupos seguem a mesma ordem fixa (A, B) das
        demais comparações; as três parcelas somam 1. Uma parcela negativa
        indica um fator que REDUZ a disparidade (age na direção oposta à
        diferença total), e uma parcela maior que 1, um fator cujo efeito
        é parcialmente compensado pelos outros. Quando a razão de FPR é
        próxima de 1, o denominador log(razão) se aproxima de zero e as
        parcelas ficam numericamente instáveis (podem assumir valores muito
        grandes); nesse caso, devem ser lidas as razões dos termos, e não as
        parcelas.
        """
        yr, yp, g = self._arrays(y_real, y_previsto, grupo)
        a, b = self._ordem(g, ordem)
        if not (g == a).any() or not (g == b).any():
            return {}
        ma = _metricas_grupo(yr[g == a], yp[g == a])
        mb = _metricas_grupo(yr[g == b], yp[g == b])

        def termos(m):
            return {
                "chances_taxa_base": m["taxa_base"] / (1 - m["taxa_base"]),
                "termo_ppv": (1 - m["ppv"]) / m["ppv"],
                "termo_sensibilidade": 1 - m["fnr"],
            }

        ta, tb = termos(ma), termos(mb)
        razoes = {k: ta[k] / tb[k] for k in ta}
        log_total = math.log(ma["fpr"] / mb["fpr"])
        resultado = {
            "grupo_a": a, "grupo_b": b,
            "fpr_a": ma["fpr"], "fpr_b": mb["fpr"], "razao_fpr": ma["fpr"] / mb["fpr"],
            "taxa_base_a": ma["taxa_base"], "taxa_base_b": mb["taxa_base"],
            "ppv_a": ma["ppv"], "ppv_b": mb["ppv"],
            "fnr_a": ma["fnr"], "fnr_b": mb["fnr"],
            "residuo_identidade": ma["fpr"] / mb["fpr"] - math.prod(razoes.values()),
        }
        for k, r in razoes.items():
            resultado[f"razao_{k}"] = r
            resultado[f"parcela_{k}"] = math.log(r) / log_total if log_total else float("nan")
        return resultado

    def comparar_pareado(self, y_real, y_previsto_a, y_proba_a, y_previsto_b, y_proba_b, grupo,
                         ordem: tuple[str, str] | None = None) -> dict:
        """
        Diferença (variante A − variante B) de AUC-ROC e das métricas de
        equidade entre grupos, com IC de 95% por bootstrap PAREADO: em
        cada reamostragem, as duas variantes são avaliadas nas mesmas
        observações. Usado no estudo de ablação para dizer se remover a
        raça ou os proxies altera o desempenho ou a disparidade além da
        variação amostral do conjunto de teste.
        """
        from sklearn.metrics import roc_auc_score

        yr = np.asarray(y_real).astype(int)
        pa, pb = np.asarray(y_previsto_a).astype(int), np.asarray(y_previsto_b).astype(int)
        qa, qb = np.asarray(y_proba_a).astype(float), np.asarray(y_proba_b).astype(float)
        g = pd.Series(grupo).astype(object).to_numpy()
        ordem = self._ordem(g, ordem)

        def medidas(idx):
            r, gg = yr[idx], g[idx]
            res = {"auc_roc": roc_auc_score(r, qa[idx]) - roc_auc_score(r, qb[idx])}
            ea = _resumo_entre_grupos({v: _metricas_grupo(r[gg == v], pa[idx][gg == v]) for v in ordem}, ordem)
            eb = _resumo_entre_grupos({v: _metricas_grupo(r[gg == v], pb[idx][gg == v]) for v in ordem}, ordem)
            for m in ("disparate_impact_ratio", "dif_fpr", "dif_fnr"):
                res[m] = ea[m] - eb[m]
            return res

        pontual = medidas(np.arange(len(yr)))
        rng = np.random.default_rng(self.semente_aleatoria)
        amostras = {k: [] for k in pontual}
        for _ in range(self.n_repeticoes_bootstrap):
            idx = rng.integers(0, len(yr), len(yr))
            if yr[idx].min() == yr[idx].max():
                continue
            for k, v in medidas(idx).items():
                if not math.isnan(v):
                    amostras[k].append(v)
        resultado = {}
        for k, v in pontual.items():
            resultado[f"delta_{k}"] = v
            resultado[f"delta_{k}_ic_inferior"] = float(np.percentile(amostras[k], 2.5)) if amostras[k] else float("nan")
            resultado[f"delta_{k}_ic_superior"] = float(np.percentile(amostras[k], 97.5)) if amostras[k] else float("nan")
        return resultado

    def avaliar_decisoes_pareado(self, y_real, y_previsto_antes, y_previsto_depois, grupo,
                                 ordem: tuple[str, str] | None = None) -> dict:
        """
        Compara duas regras de decisão sobre as MESMAS pessoas (usado no
        pós-processamento: limiar único "antes" × limiar por grupo
        "depois"). Para cada regra, devolve acurácia, recall, taxa de alto
        risco e as métricas por grupo e entre grupos (FPR, TPR, PPV,
        diferenças A − B e disparate impact), todas com IC de 95% por
        bootstrap; e as diferenças depois − antes, com IC por bootstrap
        PAREADO (as duas regras avaliadas nas mesmas reamostragens).
        """
        yr = np.asarray(y_real).astype(int)
        regras = {"antes": np.asarray(y_previsto_antes).astype(int), "depois": np.asarray(y_previsto_depois).astype(int)}
        g = pd.Series(grupo).astype(object).to_numpy()
        a, b = self._ordem(g, ordem)

        def medidas(idx):
            r, gg = yr[idx], g[idx]
            res = {}
            for etapa, yp_full in regras.items():
                p = yp_full[idx]
                geral = _metricas_grupo(r, p)
                res[f"{etapa}__acuracia"] = float((p == r).mean())
                res[f"{etapa}__recall"] = geral["tpr"]
                res[f"{etapa}__taxa_alto_risco"] = geral["taxa_alto_risco"]
                por = {v: _metricas_grupo(r[gg == v], p[gg == v]) for v in (a, b)}
                for v in (a, b):
                    for met in ("fpr", "tpr", "ppv", "taxa_alto_risco"):
                        res[f"{etapa}__{met}_{v}"] = por[v][met]
                for met, valor in _resumo_entre_grupos(por, (a, b)).items():
                    res[f"{etapa}__{met}"] = valor
            for met in ("acuracia", "recall", "disparate_impact_ratio", "dif_fpr", "dif_tpr", "dif_ppv"):
                res[f"delta__{met}"] = res[f"depois__{met}"] - res[f"antes__{met}"]
            return res

        pontual = medidas(np.arange(len(yr)))
        rng = np.random.default_rng(self.semente_aleatoria)
        amostras = {k: [] for k in pontual}
        for _ in range(self.n_repeticoes_bootstrap):
            idx = rng.integers(0, len(yr), len(yr))
            for k, v in medidas(idx).items():
                if not (isinstance(v, float) and math.isnan(v)):
                    amostras[k].append(v)
        resultado = {}
        for k, v in pontual.items():
            resultado[k] = v
            resultado[f"{k}_ic_inferior"] = float(np.percentile(amostras[k], 2.5)) if amostras[k] else float("nan")
            resultado[f"{k}_ic_superior"] = float(np.percentile(amostras[k], 97.5)) if amostras[k] else float("nan")
        return resultado

    def calcular_indice_nij(self, y_real, y_proba, grupo, grupos=("BLACK", "WHITE"), limiar: float = 0.5) -> dict:
        """
        Índice oficial de pontuação do NIJ Recidivism Forecasting
        Challenge: (1 − BS) × (1 − |FPR_BLACK − FPR_WHITE|), com as
        probabilidades binarizadas no limiar de 0,5. Aplicado aqui aos
        resultados deste trabalho apenas como medida comparativa (não é
        uma submissão à competição, que usava outros conjuntos de teste e
        outros horizontes de reincidência).
        """
        yr = np.asarray(y_real).astype(int)
        proba = np.asarray(y_proba).astype(float)
        g = pd.Series(grupo).astype(object).to_numpy()
        yp = (proba >= limiar).astype(int)
        brier = float(brier_score_loss(yr, proba))
        fprs = {v: _metricas_grupo(yr[g == v], yp[g == v])["fpr"] for v in grupos if (g == v).any()}
        if len(fprs) < 2:
            return {"brier": brier, "penalidade_equidade": float("nan"), "indice_nij": float("nan")}
        penalidade = 1 - abs(fprs[grupos[0]] - fprs[grupos[1]])
        return {"brier": brier, "penalidade_equidade": penalidade, "indice_nij": (1 - brier) * penalidade}


def equidade_em_linhas(modelo: str, atributo: str, equidade: dict, ic: dict, calibracao: dict | None = None) -> list[dict]:
    """Converte os resultados de equidade em linhas (formato longo) para CSV."""
    linhas = []
    for grupo_nome, mets in equidade["por_grupo"].items():
        for met, valor in mets.items():
            limites = ic.get(f"{grupo_nome}__{met}", {})
            linhas.append({"modelo": modelo, "atributo": atributo, "grupo": grupo_nome, "metrica": met,
                           "valor": valor, "ic_inferior": limites.get("ic_inferior"),
                           "ic_superior": limites.get("ic_superior")})
    comparacao = f"{equidade['grupo_a']}_vs_{equidade['grupo_b']}"
    for met in METRICAS_ENTRE_GRUPOS:
        limites = ic.get(met, {})
        linhas.append({"modelo": modelo, "atributo": atributo, "grupo": comparacao, "metrica": met,
                       "valor": equidade[met], "ic_inferior": limites.get("ic_inferior"),
                       "ic_superior": limites.get("ic_superior")})
    if calibracao:
        for grupo_nome, mets in calibracao.items():
            for met, valor in mets.items():
                if met == "n":
                    continue
                limites = ic.get(f"{grupo_nome}__{met}", {})
                linhas.append({"modelo": modelo, "atributo": atributo, "grupo": grupo_nome, "metrica": met,
                               "valor": valor, "ic_inferior": limites.get("ic_inferior"),
                               "ic_superior": limites.get("ic_superior")})
    return linhas
