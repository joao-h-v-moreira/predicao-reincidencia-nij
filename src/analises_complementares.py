"""
Análises complementares pedidas na revisão do orientador.

- metricas_interseccionais: equidade por raça e sexo COMBINADOS (quatro
  subgrupos: BLACK-M, BLACK-F, WHITE-M, WHITE-F), com IC por bootstrap, e
  as comparações raciais DENTRO de cada sexo (e entre sexos dentro de
  cada grupo racial). Separa a disparidade racial da disparidade por
  sexo, que é maior e se sobrepõe a ela.

- comparar_com_desafio_nij: reproduz o desenho do primeiro período do
  desafio do NIJ (treino na amostra de treino oficial, Training_Sample = 1;
  avaliação na amostra de teste oficial, Training_Sample = 0; alvo = nova
  prisão no PRIMEIRO ano, Recidivism_Arrest_Year1; só atributos
  disponíveis na soltura) e calcula as mesmas medidas da competição:
  erro de Brier separado para homens e mulheres, a média dos dois, e o
  índice de equidade e acurácia (1 − BS) × (1 − |FPR_BLACK − FPR_WHITE|)
  dentro de cada sexo, com as previsões binarizadas no limiar de 0,5.
  Assim o desempenho pode ser situado em relação aos resultados oficiais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from bias_removal import RemovedorDeVies
from fairness import ORDEM_GRUPOS, AvaliadorEquidade, _metricas_grupo
from models import GRADES_HIPERPARAMETROS, ConfiguradorAlgoritmo, ModeloBaselineDresselFarid
from preprocess import converter_alvo_binario

COLUNA_AMOSTRA_OFICIAL = "Training_Sample"
COLUNA_ALVO_ANO1 = "Recidivism_Arrest_Year1"

# Resultados oficiais do primeiro período (Year 1) do desafio, publicados
# pelo NIJ em "Recidivism Forecasting Challenge Results"
# (https://nij.ojp.gov/funding/recidivism-forecasting-challenge-results).
# Erro de Brier: quanto MENOR, melhor. Índice de equidade e acurácia:
# quanto MAIOR, melhor. Usados apenas como referência no relatório.
RESULTADOS_OFICIAIS_ANO1 = [
    {"categoria": "Equipe grande (1º lugar)", "medida": "brier_homens", "valor": 0.1900},
    {"categoria": "Equipe grande (1º lugar)", "medida": "brier_mulheres", "valor": 0.1538},
    {"categoria": "Equipe grande (1º lugar)", "medida": "brier_media", "valor": 0.1719},
    {"categoria": "Equipe pequena (1º lugar)", "medida": "brier_homens", "valor": 0.1916},
    {"categoria": "Equipe pequena (1º lugar)", "medida": "brier_mulheres", "valor": 0.1542},
    {"categoria": "Equipe pequena (1º lugar)", "medida": "brier_media", "valor": 0.1733},
    {"categoria": "Viés racial (1º lugar)", "medida": "indice_equidade_homens", "valor": 0.8090},
    {"categoria": "Viés racial (1º lugar)", "medida": "indice_equidade_mulheres", "valor": 0.8446},
]


def _rotulo_interseccional(raca: pd.Series, sexo: pd.Series) -> pd.Series:
    combinado = raca.astype(object).astype(str) + "_" + sexo.astype(object).astype(str)
    return combinado.where(raca.notna() & sexo.notna())


def metricas_interseccionais(modelo: str, y_real, y_previsto, y_proba, raca, sexo,
                             aval_e: AvaliadorEquidade) -> pd.DataFrame:
    """Linhas (formato longo): métricas por subgrupo raça × sexo e comparações dentro de cada estrato."""
    y_real = pd.Series(np.asarray(y_real).astype(int), index=raca.index)
    y_previsto = pd.Series(np.asarray(y_previsto).astype(int), index=raca.index)
    y_proba = pd.Series(np.asarray(y_proba, dtype=float), index=raca.index)
    grupo = _rotulo_interseccional(raca, sexo)
    linhas = []

    # (1) os quatro subgrupos
    ordem = ("BLACK_M", "WHITE_M")
    eq = aval_e.calcular_equidade(y_real, y_previsto, grupo, ordem)
    ic = aval_e.calcular_intervalo_confianca_bootstrap(y_real, y_previsto, grupo, y_proba, ordem)
    cal = aval_e.calcular_calibracao_por_grupo(y_real, y_proba, grupo)
    for g, mets in eq["por_grupo"].items():
        for met in ("n", "taxa_base", "taxa_alto_risco", "fpr", "fnr", "ppv"):
            lim = ic.get(f"{g}__{met}", {})
            linhas.append({"modelo": modelo, "recorte": "subgrupo", "grupo": g, "metrica": met, "valor": mets[met],
                           "ic_inferior": lim.get("ic_inferior"), "ic_superior": lim.get("ic_superior")})
        for met in ("calibracao_no_agregado", "brier_skill"):
            lim = ic.get(f"{g}__{met}", {})
            linhas.append({"modelo": modelo, "recorte": "subgrupo", "grupo": g, "metrica": met,
                           "valor": cal[g][met], "ic_inferior": lim.get("ic_inferior"),
                           "ic_superior": lim.get("ic_superior")})

    # (2) raça dentro de cada sexo e (3) sexo dentro de cada grupo racial
    estratos = [("raca", "sexo", s, sexo == s) for s in ("M", "F")] + \
               [("sexo", "raca", r, raca == r) for r in ("BLACK", "WHITE")]
    for atributo, estrato_nome, estrato_valor, mascara in estratos:
        mascara = mascara.fillna(False).to_numpy()
        serie_grupo = (raca if atributo == "raca" else sexo)[mascara]
        ordem_a = ORDEM_GRUPOS[atributo]
        eq_s = aval_e.calcular_equidade(y_real[mascara], y_previsto[mascara], serie_grupo, ordem_a)
        ic_s = aval_e.calcular_intervalo_confianca_bootstrap(y_real[mascara], y_previsto[mascara], serie_grupo,
                                                             None, ordem_a)
        recorte = f"{ordem_a[0]}_vs_{ordem_a[1]} | {estrato_nome}={estrato_valor}"
        for met in ("disparate_impact_ratio", "diferenca_paridade_demografica", "dif_fpr", "dif_fnr", "dif_ppv"):
            lim = ic_s.get(met, {})
            linhas.append({"modelo": modelo, "recorte": recorte, "grupo": f"{ordem_a[0]}_vs_{ordem_a[1]}",
                           "metrica": met, "valor": eq_s[met], "ic_inferior": lim.get("ic_inferior"),
                           "ic_superior": lim.get("ic_superior")})
    return pd.DataFrame(linhas)


def dif_fpr_racial_por_sexo(y_real, y_previsto, raca, sexo) -> dict:
    """Diferença de FPR BLACK − WHITE dentro de cada sexo (sem IC; usada nas repetições)."""
    yr, yp = np.asarray(y_real).astype(int), np.asarray(y_previsto).astype(int)
    r, s = np.asarray(raca, dtype=object), np.asarray(sexo, dtype=object)
    saida = {}
    for sx, nome in (("M", "homens"), ("F", "mulheres")):
        m = s == sx
        fpr = {g: _metricas_grupo(yr[m & (r == g)], yp[m & (r == g)])["fpr"] for g in ("BLACK", "WHITE")}
        saida[f"raca_dif_fpr_{nome}"] = fpr["BLACK"] - fpr["WHITE"]
    return saida


def _medidas_desafio(y, proba, raca, sexo, limiar: float = 0.5) -> dict:
    y, proba = np.asarray(y).astype(int), np.asarray(proba, dtype=float)
    r, s = np.asarray(raca, dtype=object), np.asarray(sexo, dtype=object)
    yp = (proba >= limiar).astype(int)
    res = {"n_teste": int(len(y)), "taxa_base": float(y.mean()),
           "auc_roc": float(roc_auc_score(y, proba)), "brier": float(brier_score_loss(y, proba))}
    for sx, nome in (("M", "homens"), ("F", "mulheres")):
        m = s == sx
        bs = float(brier_score_loss(y[m], proba[m]))
        fpr = {g: _metricas_grupo(y[m & (r == g)], yp[m & (r == g)])["fpr"] for g in ("BLACK", "WHITE")}
        res[f"n_{nome}"] = int(m.sum())
        res[f"auc_roc_{nome}"] = float(roc_auc_score(y[m], proba[m])) if len(np.unique(y[m])) > 1 else float("nan")
        res[f"brier_{nome}"] = bs
        res[f"dif_fpr_racial_{nome}"] = fpr["BLACK"] - fpr["WHITE"]
        res[f"indice_equidade_{nome}"] = (1 - bs) * (1 - abs(fpr["BLACK"] - fpr["WHITE"]))
    res["brier_media"] = (res["brier_homens"] + res["brier_mulheres"]) / 2
    res["indice_equidade_media"] = (res["indice_equidade_homens"] + res["indice_equidade_mulheres"]) / 2
    return res


def comparar_com_desafio_nij(df_bruto: pd.DataFrame, x_soltura: pd.DataFrame, raca: pd.Series, sexo: pd.Series,
                             algoritmos: list[str], cv_interno) -> pd.DataFrame | None:
    """
    Reproduz o primeiro período do desafio (ver docstring do módulo).
    `x_soltura` já vem sem a raça e sem as variáveis do período de
    supervisão (e sem os alvos por ano). Devolve None se o arquivo não
    tiver as colunas da divisão oficial.
    """
    if COLUNA_AMOSTRA_OFICIAL not in df_bruto.columns or COLUNA_ALVO_ANO1 not in df_bruto.columns:
        return None
    oficial = pd.to_numeric(df_bruto[COLUNA_AMOSTRA_OFICIAL], errors="coerce").loc[x_soltura.index]
    idx_tr, idx_te = x_soltura.index[oficial == 1], x_soltura.index[oficial == 0]
    y1 = converter_alvo_binario(df_bruto[COLUNA_ALVO_ANO1].loc[x_soltura.index])
    y_tr, y_te = y1.loc[idx_tr], y1.loc[idx_te]

    removedor = RemovedorDeVies()
    proxies = (removedor.identificar_proxies_categoricas(x_soltura.loc[idx_tr], raca.loc[idx_tr])
               + removedor.identificar_proxies_numericas(x_soltura.loc[idx_tr], raca.loc[idx_tr]))
    x = removedor.aplicar_remocao(x_soltura, proxies)
    linhas = []
    for nome in algoritmos:
        cfg = ConfiguradorAlgoritmo(nome)
        hp = cfg.buscar_hiperparametros(x.loc[idx_tr], y_tr, GRADES_HIPERPARAMETROS[nome], cv_interno)
        cfg.definir_hiperparametros(hp)
        modelo = cfg.instanciar()
        modelo.treinar(x.loc[idx_tr], y_tr)
        proba = modelo.prever_probabilidade(x.loc[idx_te])
        linhas.append({"origem": "este trabalho", "modelo": nome, **_medidas_desafio(y_te, proba, raca.loc[idx_te],
                                                                                   sexo.loc[idx_te]),
                       "proxies_removidas": ";".join(proxies) or "-"})
    base = ModeloBaselineDresselFarid()
    xb = base.preparar_atributos(df_bruto.loc[x_soltura.index])
    base.treinar(xb.loc[idx_tr], y_tr)
    linhas.append({"origem": "este trabalho", "modelo": "baseline_dressel_farid",
                   **_medidas_desafio(y_te, base.prever_probabilidade(xb.loc[idx_te]), raca.loc[idx_te],
                                      sexo.loc[idx_te]), "proxies_removidas": "-"})
    # modelo ingênuo: prevê para cada pessoa a taxa de reincidência do seu sexo na amostra de treino
    taxa_sexo = y_tr.groupby(sexo.loc[idx_tr]).mean()
    ingenuo = sexo.loc[idx_te].map(taxa_sexo).fillna(float(y_tr.mean())).to_numpy()
    linhas.append({"origem": "este trabalho", "modelo": "ingenuo_taxa_base_por_sexo",
                   **_medidas_desafio(y_te, ingenuo, raca.loc[idx_te], sexo.loc[idx_te]), "proxies_removidas": "-"})
    tabela = pd.DataFrame(linhas)
    tabela["n_treino"] = int(len(idx_tr))
    oficiais = pd.DataFrame(RESULTADOS_OFICIAIS_ANO1).pivot(index="categoria", columns="medida", values="valor")
    oficiais = oficiais.reset_index().rename(columns={"categoria": "modelo"})
    oficiais.insert(0, "origem", "NIJ - resultados oficiais do ano 1")
    return pd.concat([tabela, oficiais], ignore_index=True)
