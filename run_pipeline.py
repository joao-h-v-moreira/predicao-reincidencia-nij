"""
Script principal do pipeline de predição de reincidência criminal.

Orquestra todas as etapas (UC01-UC10), publicando eventos de auditoria
(RNF11) a cada etapa concluída.

Desenho experimental (Seção 2.5 do TCC):
- Pergunta: no momento da soltura, qual o risco de nova prisão em até 3
  anos? A análise principal usa apenas atributos disponíveis na soltura
  (conjunto "soltura"); o conjunto "completo" (com as variáveis medidas
  durante a condicional) entra só como análise de sensibilidade.
- Validação aninhada: laço externo com N divisões treino/teste (70/30)
  aleatórias e estratificadas (sementes 42, 43, ...); laço interno com
  k-fold (k = 3) dentro do treino para a busca de hiperparâmetros. A
  primeira divisão (semente 42) é a divisão de referência, usada nas
  figuras e nas análises detalhadas (ICs por bootstrap, equidade por
  grupo, decomposição, ablação, proxy multivariado, pós-processamento).
- Pré-processamento sempre ajustado só no treino (Pipeline scikit-learn).
- Proxies bivariados identificados só no treino de cada divisão.

Uso:
    python run_pipeline.py --dataset caminho/para/full_dataset.csv
    python run_pipeline.py --dataset ... --repeticoes 3   # execução mais curta
    python run_pipeline.py --sintetico --repeticoes 2      # smoke test, sem dados reais
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ablation import VARIANTES, AvaliadorAblacao
from bias_removal import RemovedorDeVies
from comparison import ComparadorModelos
from dataset import COLUNA_ALVO_PADRAO, COLUNA_RACA_PADRAO, COLUNAS_CATEGORICAS_CODIFICADAS_NUMERICAMENTE, Dataset
from eda import AnalisadorExploratorio
from evaluation import AvaliadorDesempenho, metricas_com_ic_em_colunas, resumir_repeticoes
from events import RegistradorDeEventos
from fairness import ORDEM_GRUPOS, AvaliadorEquidade, equidade_em_linhas
from models import GRADES_HIPERPARAMETROS, ConfiguradorAlgoritmo, ModeloBaselineDresselFarid
from postprocessing import ajustar_limiar_por_grupo, dividir_calibracao_avaliacao
from preprocess import Preprocessador
from proxy_multivariado import detectar_proxy_multivariado
from report import RelatorioResultados
from split import DivisorDados
from synthetic_data import gerar_dataset_sintetico
from visualizacao import plotar_calibracao_por_grupo, plotar_curvas_calibracao, plotar_curvas_roc

ALGORITMOS = ["regressao_logistica", "random_forest", "gradient_boosting"]
BASELINE = "baseline_dressel_farid"
ATRIBUTOS_PROTEGIDOS = ["raca", "sexo"]
COLUNA_SEXO = "Gender"
MODELO_DETALHADO = "gradient_boosting"


def _log(msg: str) -> None:
    print(f"\n*** [{time.strftime('%H:%M:%S')}] {msg} ***", flush=True)


def _versoes_ambiente() -> dict:
    import fairlearn, matplotlib, scipy, sklearn
    return {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "fairlearn": fairlearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _identificar_proxies_no_treino(x_treino: pd.DataFrame, raca_treino: pd.Series):
    removedor = RemovedorDeVies()
    diagnostico = removedor.diagnostico_associacao(x_treino, raca_treino)
    proxies = (removedor.identificar_proxies_categoricas(x_treino, raca_treino)
               + removedor.identificar_proxies_numericas(x_treino, raca_treino))
    return proxies, diagnostico, removedor


def _treinar_algoritmo(nome, x_tr, y_tr, cv_interno, hiperparametros=None):
    cfg = ConfiguradorAlgoritmo(nome)
    if hiperparametros is None and nome in GRADES_HIPERPARAMETROS:
        hiperparametros = cfg.buscar_hiperparametros(x_tr, y_tr, GRADES_HIPERPARAMETROS[nome], cv_interno)
    cfg.definir_hiperparametros(hiperparametros or {})
    modelo = cfg.instanciar()
    modelo.treinar(x_tr, y_tr)
    return modelo, (hiperparametros or {})


def _avaliar_detalhado(nome, y_te, y_prev, y_proba, grupos_teste: dict, aval_d, aval_e, hiperparametros, semente: int):
    """Avaliação completa (com ICs) de um modelo na divisão de referência."""
    metricas = aval_d.calcular_metricas(y_te, y_prev, y_proba)
    ic = aval_d.calcular_intervalo_confianca_bootstrap(y_te, y_prev, y_proba)
    registro = {"modelo": nome, "semente_divisao": semente, "n_teste": int(len(y_te)),
                **metricas_com_ic_em_colunas(metricas, ic)}
    linhas, decomposicoes = [], []
    for atributo, grupo in grupos_teste.items():
        ordem = ORDEM_GRUPOS[atributo]
        eq = aval_e.calcular_equidade(y_te, y_prev, grupo, ordem)
        cal = aval_e.calcular_calibracao_por_grupo(y_te, y_proba, grupo)
        ic_eq = aval_e.calcular_intervalo_confianca_bootstrap(y_te, y_prev, grupo, y_proba, ordem)
        linhas += equidade_em_linhas(nome, atributo, eq, ic_eq, cal)
        for met in ("disparate_impact_ratio", "dif_fpr", "dif_fnr"):
            registro[f"{atributo}_{met}"] = eq[met]
            registro[f"{atributo}_{met}_ic_inferior"] = ic_eq.get(met, {}).get("ic_inferior")
            registro[f"{atributo}_{met}_ic_superior"] = ic_eq.get(met, {}).get("ic_superior")
        dec = aval_e.decompor_disparidade_fpr(y_te, y_prev, grupo, ordem)
        if dec:
            decomposicoes.append({"modelo": nome, "atributo": atributo, **dec})
    nij = aval_e.calcular_indice_nij(y_te, y_proba, grupos_teste["raca"])
    registro["penalidade_equidade_nij"] = nij["penalidade_equidade"]
    registro["indice_nij"] = nij["indice_nij"]
    registro["hiperparametros"] = json.dumps(hiperparametros, ensure_ascii=False)
    return registro, linhas, decomposicoes


def _avaliar_rapido(nome, y_te, y_prev, y_proba, grupos_teste: dict, aval_d, aval_e) -> dict:
    """Avaliação sem ICs, usada em cada repetição do laço externo."""
    m = aval_d.calcular_metricas(y_te, y_prev, y_proba)
    linha = {"modelo": nome, **{k: m[k] for k in ("acuracia", "auc_roc", "auc_pr", "brier", "recall", "precisao")}}
    for atributo, grupo in grupos_teste.items():
        eq = aval_e.calcular_equidade(y_te, y_prev, grupo, ORDEM_GRUPOS[atributo])
        for met in ("disparate_impact_ratio", "dif_fpr", "dif_fnr"):
            linha[f"{atributo}_{met}"] = eq[met]
    linha["indice_nij"] = aval_e.calcular_indice_nij(y_te, y_proba, grupos_teste["raca"])["indice_nij"]
    return linha


def _resumo_textual(n_registros, n_teste, proxies, comparacao, pares, resumo_rep, equidade, decomposicao,
                    ablacao, pos, sens, proxy_mv, duracao) -> str:
    """Resumo em texto corrido dos principais resultados (RNF05). Valores com IC de 95% entre colchetes."""
    def f(x, nd=3):
        return f"{x:.{nd}f}".replace(".", ",")

    def ic(lo, hi, nd=3):
        return f"[{f(lo, nd)}; {f(hi, nd)}]"

    def eq(atributo, grupo, metrica):
        r = equidade[(equidade.modelo == MODELO_DETALHADO) & (equidade.atributo == atributo)
                     & (equidade.grupo == grupo) & (equidade.metrica == metrica)].iloc[0]
        return r.valor, r.ic_inferior, r.ic_superior

    c = comparacao.set_index("modelo")
    linhas = ["RESUMO DOS RESULTADOS", "=" * 21, "",
              f"Registros: {n_registros}; conjunto de teste da divisão de referência (semente 42): {n_teste}.",
              f"Proxies removidos (teste bivariado no treino): {', '.join(proxies) or 'nenhum'}.", "",
              "Desempenho na divisão de referência (AUC-ROC [IC 95%]; Brier):"]
    for m, r in c.iterrows():
        linhas.append(f"  - {m}: {f(r.auc_roc)} {ic(r.auc_roc_ic_inferior, r.auc_roc_ic_superior)}; Brier {f(r.brier)}")
    linhas.append("Diferenças pareadas de AUC-ROC (A − B):")
    for r in pares.itertuples():
        linhas.append(f"  - {r.modelo_a} − {r.modelo_b}: {f(r.diferenca_auc, 4)} {ic(r.ic_inferior, r.ic_superior, 4)}")
    linhas.append("AUC-ROC nas repetições do laço externo (média ± desvio-padrão):")
    for m, r in resumo_rep.set_index("modelo").iterrows():
        linhas.append(f"  - {m}: {f(r.auc_roc_media)} ± {f(r.auc_roc_dp)} ({int(r.n_repeticoes)} divisões)")
    linhas += ["", f"Equidade do modelo detalhado ({MODELO_DETALHADO}):"]
    for atributo, comp in (("raca", "BLACK_vs_WHITE"), ("sexo", "M_vs_F")):
        di, dfpr, dfnr = (eq(atributo, comp, k) for k in ("disparate_impact_ratio", "dif_fpr", "dif_fnr"))
        linhas.append(f"  - {comp}: disparate impact {f(di[0])} {ic(di[1], di[2])}; "
                      f"dif. FPR {f(dfpr[0])} {ic(dfpr[1], dfpr[2])}; dif. FNR {f(dfnr[0])} {ic(dfnr[1], dfnr[2])}")
    dr = decomposicao[(decomposicao.modelo == MODELO_DETALHADO) & (decomposicao.atributo == "raca")].iloc[0]
    linhas.append(f"  - Decomposição da razão de FPR por raça: taxa-base {f(dr.parcela_chances_taxa_base, 2)}, "
                  f"precisão {f(dr.parcela_termo_ppv, 2)}, sensibilidade {f(dr.parcela_termo_sensibilidade, 2)}")
    linhas.append(f"  - Reconstrução da raça a partir dos atributos finais: AUC-ROC {f(proxy_mv['auc_roc_reconstrucao_raca'])} "
                  f"(efeito {proxy_mv['tamanho_efeito_rice_harris']})")
    linhas += ["", "Ablação (variação em relação à variante com raça e com proxies):"]
    for v, r in ablacao.set_index("variante").iterrows():
        linhas.append(f"  - {v}: Δ AUC {f(r.delta_auc_roc, 4)} {ic(r.delta_auc_roc_ic_inferior, r.delta_auc_roc_ic_superior, 4)}; "
                      f"Δ dif. FPR {f(r.delta_dif_fpr)} {ic(r.delta_dif_fpr_ic_inferior, r.delta_dif_fpr_ic_superior)}")
    pa, pd_, dd = pos.set_index("etapa").loc["antes"], pos.set_index("etapa").loc["depois"], \
        pos.set_index("etapa").loc["diferenca_depois_menos_antes"]
    linhas += ["", "Pós-processamento por limiar por grupo (metade de avaliação do teste):",
               f"  - dif. FPR: {f(pa.dif_fpr)} -> {f(pd_.dif_fpr)} {ic(pd_.dif_fpr_ic_inferior, pd_.dif_fpr_ic_superior)}; "
               f"redução pareada {f(dd.delta_dif_fpr)} {ic(dd.delta_dif_fpr_ic_inferior, dd.delta_dif_fpr_ic_superior)}",
               f"  - recall: {f(pa.recall_geral)} -> {f(pd_.recall_geral)}; acurácia: {f(pa.acuracia)} -> {f(pd_.acuracia)}"]
    s1 = sens.set_index("conjunto_atributos").loc["completo"]
    linhas += ["", "Sensibilidade (conjunto completo, com variáveis do período de supervisão):",
               f"  - Δ AUC-ROC em relação ao conjunto da soltura: {f(s1.diferenca_auc_vs_soltura)} "
               f"{ic(s1.diferenca_auc_ic_inferior, s1.diferenca_auc_ic_superior)}",
               "", f"Duração da execução: {f(duracao, 1)} minutos."]
    return "\n".join(linhas) + "\n"


def executar_pipeline(caminho_csv: str, usar_sintetico: bool, resultados_dir: str = "results", n_repeticoes: int = 10):
    inicio = time.time()
    Path(resultados_dir).mkdir(parents=True, exist_ok=True)
    registrador = RegistradorDeEventos(f"{resultados_dir}/eventos.jsonl")
    ambiente = _versoes_ambiente()
    Path(f"{resultados_dir}/ambiente_execucao.json").write_text(json.dumps(ambiente, indent=2, ensure_ascii=False), encoding="utf-8")
    registrador.publicar("ambiente_registrado", ambiente)

    # ---------------- UC01: importar dataset ----------------
    if usar_sintetico:
        _log("MODO SMOKE TEST: dataset SINTÉTICO (não é o NIJ real)")
        df_bruto = gerar_dataset_sintetico()
        for col in COLUNAS_CATEGORICAS_CODIFICADAS_NUMERICAMENTE:
            if col in df_bruto.columns:
                df_bruto[col] = df_bruto[col].astype("string")
    else:
        df_bruto = Dataset(caminho_csv).importar()
    registrador.publicar("dataset_importado", {"n_linhas": len(df_bruto), "n_colunas": df_bruto.shape[1]})

    # ---------------- UC02: EDA ----------------
    eda = AnalisadorExploratorio(df_bruto, COLUNA_ALVO_PADRAO)
    eda.gerar_estatisticas_descritivas()
    ausentes = eda.calcular_valores_ausentes()
    pd.DataFrame({"variavel": list(ausentes), "proporcao_ausente": list(ausentes.values())}).to_csv(
        f"{resultados_dir}/valores_ausentes.csv", index=False)
    # ausentes por sexo: verifica se a ausência de alguma variável se concentra
    # num dos sexos (nesse caso, a categoria "Ausente" passa a codificar o sexo)
    com_ausentes = [c for c in df_bruto.columns if df_bruto[c].isna().any() and c != COLUNA_SEXO]
    tabela_ausentes_sexo = (df_bruto[com_ausentes].isna().groupby(df_bruto[COLUNA_SEXO]).mean().T
                            .add_prefix("proporcao_ausente_").reset_index().rename(columns={"index": "variavel"}))
    tabela_ausentes_sexo.to_csv(f"{resultados_dir}/valores_ausentes_por_sexo.csv", index=False)
    caminho_eda = eda.plotar_distribuicoes(f"{resultados_dir}/eda_distribuicoes.png")
    registrador.publicar("eda_concluida", {"figura": str(caminho_eda)})

    # ---------------- UC03/UC04: separar alvo, raça e atributos ----------------
    preproc = Preprocessador()
    x_soltura, y, raca = preproc.preparar_atributos(df_bruto, COLUNA_ALVO_PADRAO, COLUNA_RACA_PADRAO, "soltura")
    x_completo, _, _ = preproc.preparar_atributos(df_bruto, COLUNA_ALVO_PADRAO, COLUNA_RACA_PADRAO, "completo")
    sexo = df_bruto[COLUNA_SEXO].astype(object).map(lambda v: str(v) if pd.notna(v) else np.nan)
    registrador.publicar("dados_preprocessados", {
        "n_atributos_soltura": x_soltura.shape[1], "n_atributos_completo": x_completo.shape[1],
        "atributos_soltura": list(x_soltura.columns),
    })

    taxas_base = {"geral": float(y.mean())}
    for nome_grupo, serie in (("raca", raca), ("sexo", sexo)):
        serie = serie.loc[y.index]
        for g in sorted(serie.dropna().unique()):
            taxas_base[f"{nome_grupo}_{g}"] = float(y[serie == g].mean())
    pd.DataFrame([taxas_base]).to_csv(f"{resultados_dir}/taxas_base_reincidencia.csv", index=False)

    divisor = DivisorDados(proporcao_teste=0.3, semente_aleatoria=42)
    sementes, cv_interno = divisor.validacao_cruzada_aninhada(n_repeticoes=n_repeticoes, k_interno=3)
    aval_d = AvaliadorDesempenho(n_repeticoes_bootstrap=1000)
    aval_e = AvaliadorEquidade(ATRIBUTOS_PROTEGIDOS, n_repeticoes_bootstrap=1000)

    # =====================================================================
    # DIVISÃO DE REFERÊNCIA (semente 42): análises detalhadas
    # =====================================================================
    semente_ref = sementes[0]
    x_tr0, x_te0, y_tr, y_te = divisor.dividir_treino_teste(x_soltura, y, semente=semente_ref)
    idx_tr, idx_te = x_tr0.index, x_te0.index

    proxies, diagnostico, removedor = _identificar_proxies_no_treino(x_tr0, raca.loc[idx_tr])
    x_final = removedor.aplicar_remocao(x_soltura, proxies)
    x_tr, x_te = x_final.loc[idx_tr], x_final.loc[idx_te]
    grupos_te = {"raca": raca.loc[idx_te], "sexo": sexo.loc[idx_te]}
    _log(f"Proxies identificados no treino e removidos: {proxies or 'nenhum'}")
    registrador.publicar("proxies_identificados", {
        "proxies": proxies, "limiar_cramer_v": removedor.limiar_cramer_v,
        "limiar_correlacao_numerica": removedor.limiar_correlacao_numerica, "n_treino": len(idx_tr),
    })

    registros, linhas_eq, decomposicoes = [], [], []
    probas, modelos, hiper_ref = {}, {}, {}
    for nome in ALGORITMOS:
        _log(f"Divisão de referência: busca de hiperparâmetros e treino de {nome}")
        modelo, hp = _treinar_algoritmo(nome, x_tr, y_tr, cv_interno)
        modelos[nome], hiper_ref[nome] = modelo, hp
        registrador.publicar("modelo_treinado", {"modelo": nome, "hiperparametros": hp})
        y_prev, y_proba = modelo.prever(x_te), modelo.prever_probabilidade(x_te)
        probas[nome] = y_proba
        reg, linhas, dec = _avaliar_detalhado(nome, y_te, y_prev, y_proba, grupos_te, aval_d, aval_e, hp, semente_ref)
        registros.append(reg)
        linhas_eq += linhas
        decomposicoes += dec
        registrador.publicar("avaliacao_concluida", {"modelo": nome, "auc_roc": reg["auc_roc"]})

    baseline = ModeloBaselineDresselFarid()
    x_base = baseline.preparar_atributos(df_bruto)
    baseline.treinar(x_base.loc[idx_tr], y_tr)
    registrador.publicar("modelo_treinado", {"modelo": BASELINE})
    y_prev_b, y_proba_b = baseline.prever(x_base.loc[idx_te]), baseline.prever_probabilidade(x_base.loc[idx_te])
    probas[BASELINE] = y_proba_b
    reg, linhas, dec = _avaliar_detalhado(BASELINE, y_te, y_prev_b, y_proba_b, grupos_te, aval_d, aval_e, {}, semente_ref)
    registros.append(reg)
    linhas_eq += linhas
    decomposicoes += dec
    registrador.publicar("avaliacao_concluida", {"modelo": BASELINE, "auc_roc": reg["auc_roc"]})

    tabela_comparacao = ComparadorModelos().comparar(registros)
    registrador.publicar("comparacao_concluida", {"n_modelos": len(registros)})
    tabela_equidade = pd.DataFrame(linhas_eq)
    tabela_decomposicao = pd.DataFrame(decomposicoes)

    _log("Comparação pareada de AUC-ROC entre modelos (bootstrap pareado)")
    pares = [("gradient_boosting", "random_forest"), ("gradient_boosting", "regressao_logistica"),
             ("random_forest", "regressao_logistica")] + [(m, BASELINE) for m in ALGORITMOS]
    tabela_pares = pd.DataFrame([
        {"modelo_a": a, "modelo_b": b, **aval_d.comparar_auc_pareado(y_te, probas[a], probas[b])} for a, b in pares
    ])

    plotar_curvas_roc(y_te, probas, f"{resultados_dir}/curvas_roc.png")
    plotar_curvas_calibracao(y_te, probas, f"{resultados_dir}/curvas_calibracao.png")
    plotar_calibracao_por_grupo(y_te, probas[MODELO_DETALHADO], grupos_te["raca"], MODELO_DETALHADO,
                                f"{resultados_dir}/calibracao_por_grupo_raca.png")
    registrador.publicar("graficos_gerados", {"arquivos": ["curvas_roc.png", "curvas_calibracao.png", "calibracao_por_grupo_raca.png"]})

    _log("Importância por permutação dos atributos do Gradient Boosting (conjunto de teste)")
    perm = permutation_importance(modelos[MODELO_DETALHADO].pipeline, x_te, y_te, scoring="roc_auc",
                                  n_repeats=5, random_state=42, n_jobs=-1)
    tabela_importancia = pd.DataFrame({
        "variavel": x_te.columns, "queda_auc_media": perm.importances_mean, "queda_auc_dp": perm.importances_std,
    }).sort_values("queda_auc_media", ascending=False)

    # ---------------- Estudo de ablação (RF11) ----------------
    # Todas as variantes usam os MESMOS hiperparâmetros do Gradient
    # Boosting selecionados na divisão de referência, para isolar o efeito
    # de incluir/excluir raça e proxies. A variante "sem raça, sem
    # proxies" é, por construção, idêntica ao modelo principal.
    _log("Estudo de ablação (hiperparâmetros fixos do Gradient Boosting de referência)")
    ablacao = AvaliadorAblacao()
    ordem_raca = ORDEM_GRUPOS["raca"]
    previsoes_ablacao = {}
    for com_raca in (True, False):
        for com_proxies in (True, False):
            variante = f"{'com' if com_raca else 'sem'}_raca_{'com' if com_proxies else 'sem'}_proxies"
            x_var = x_soltura if com_proxies else removedor.aplicar_remocao(x_soltura, proxies)
            if com_raca:
                x_var = x_var.assign(**{COLUNA_RACA_PADRAO: raca})
            modelo_var, _ = _treinar_algoritmo(MODELO_DETALHADO, x_var.loc[idx_tr], y_tr, cv_interno, hiper_ref[MODELO_DETALHADO])
            yp, ypb = modelo_var.prever(x_var.loc[idx_te]), modelo_var.prever_probabilidade(x_var.loc[idx_te])
            previsoes_ablacao[variante] = (yp, ypb)
            d = aval_d.calcular_metricas(y_te, yp, ypb)
            ic_d = aval_d.calcular_intervalo_confianca_bootstrap(y_te, yp, ypb)
            d["auc_roc_ic_inferior"] = ic_d["auc_roc"]["ic_inferior"]
            d["auc_roc_ic_superior"] = ic_d["auc_roc"]["ic_superior"]
            eq = aval_e.calcular_equidade(y_te, yp, grupos_te["raca"], ordem_raca)
            ic_eq = aval_e.calcular_intervalo_confianca_bootstrap(y_te, yp, grupos_te["raca"], None, ordem_raca)
            res_eq = {}
            for met in ("disparate_impact_ratio", "dif_fpr", "dif_fnr"):
                res_eq[met] = eq[met]
                res_eq[f"{met}_ic_inferior"] = ic_eq.get(met, {}).get("ic_inferior")
                res_eq[f"{met}_ic_superior"] = ic_eq.get(met, {}).get("ic_superior")
            # diferença pareada em relação à variante (i) "com raça, com proxies"
            yp0, ypb0 = previsoes_ablacao["com_raca_com_proxies"]
            res_eq.update(aval_e.comparar_pareado(y_te, yp, ypb, yp0, ypb0, grupos_te["raca"], ordem_raca))
            res_eq["proxies_removidas"] = ";".join(proxies) if (not com_proxies and proxies) else "-"
            ablacao.executar_ablacao(variante, d, res_eq)
    tabela_ablacao = ablacao.comparar_variantes()
    registrador.publicar("ablacao_concluida", {"n_variantes": len(VARIANTES)})

    # ---------------- Proxy multivariado ----------------
    _log("Diagnóstico de proxy multivariado (conjunto da soltura, sem raça e sem proxies)")
    resumo_pm, importancia_pm = detectar_proxy_multivariado(x_final, raca)
    resumo_pm["conjunto_atributos"] = "soltura"
    registrador.publicar("proxy_multivariado_concluido", resumo_pm)

    # ---------------- Pós-processamento (Hardt, Price e Srebro, 2016) ----------------
    _log("Pós-processamento por limiar por grupo (usa a raça no momento da decisão)")
    raca_te = grupos_te["raca"]
    x_cal, y_cal, r_cal, x_av, y_av, r_av = dividir_calibracao_avaliacao(x_te, y_te, raca_te)
    pipe_gb = modelos[MODELO_DETALHADO].pipeline
    y_antes = pipe_gb.predict(x_av)
    y_depois, _ = ajustar_limiar_por_grupo(pipe_gb, x_cal, y_cal, r_cal, x_av, r_av)
    linhas_pp = []
    for etapa, yp in (("antes", y_antes), ("depois", y_depois)):
        eq = aval_e.calcular_equidade(y_av, yp, r_av, ordem_raca)
        ic = aval_e.calcular_intervalo_confianca_bootstrap(y_av, yp, r_av, None, ordem_raca)
        linha = {"etapa": etapa, "n_avaliacao": len(y_av),
                 "acuracia": float((np.asarray(yp) == np.asarray(y_av)).mean()),
                 "recall_geral": float(((np.asarray(yp) == 1) & (np.asarray(y_av) == 1)).sum() / max((np.asarray(y_av) == 1).sum(), 1)),
                 "taxa_alto_risco_geral": float(np.mean(yp))}
        for met in ("disparate_impact_ratio", "dif_fpr", "dif_tpr", "dif_ppv"):
            linha[met] = eq[met]
            linha[f"{met}_ic_inferior"] = ic.get(met, {}).get("ic_inferior")
            linha[f"{met}_ic_superior"] = ic.get(met, {}).get("ic_superior")
        for g, mets in eq["por_grupo"].items():
            for met in ("fpr", "tpr", "ppv", "taxa_alto_risco"):
                linha[f"{met}_{g}"] = mets[met]
        linhas_pp.append(linha)
    tabela_pos = pd.DataFrame(linhas_pp)
    # diferença pareada depois − antes, nas mesmas observações de avaliação
    ypb_av = pipe_gb.predict_proba(x_av)[:, 1]
    delta_pp = aval_e.comparar_pareado(y_av, y_depois, ypb_av, y_antes, ypb_av, r_av, ordem_raca)
    tabela_pos = pd.concat([tabela_pos, pd.DataFrame([{
        "etapa": "diferenca_depois_menos_antes", "n_avaliacao": len(y_av),
        **{k: v for k, v in delta_pp.items() if not k.startswith("delta_auc_roc")}}])], ignore_index=True)
    registrador.publicar("pos_processamento_concluido", {"n_calibracao": len(y_cal), "n_avaliacao": len(y_av)})

    # ---------------- Sensibilidade: conjunto completo de atributos ----------------
    _log("Análise de sensibilidade: conjunto completo (inclui variáveis do período de supervisão)")
    xc_tr0 = x_completo.loc[idx_tr]
    proxies_c, diagnostico_c, removedor_c = _identificar_proxies_no_treino(xc_tr0, raca.loc[idx_tr])
    xc_final = removedor_c.aplicar_remocao(x_completo, proxies_c)
    modelo_c, hp_c = _treinar_algoritmo(MODELO_DETALHADO, xc_final.loc[idx_tr], y_tr, cv_interno)
    yp_c, ypb_c = modelo_c.prever(xc_final.loc[idx_te]), modelo_c.prever_probabilidade(xc_final.loc[idx_te])
    reg_c, _, dec_c = _avaliar_detalhado(MODELO_DETALHADO, y_te, yp_c, ypb_c, grupos_te, aval_d, aval_e, hp_c, semente_ref)
    resumo_pm_c, _ = detectar_proxy_multivariado(xc_final, raca, calcular_importancias=False)
    resumo_pm_c["conjunto_atributos"] = "completo"
    reg_s = next(r for r in registros if r["modelo"] == MODELO_DETALHADO)
    dec_s = next(d for d in decomposicoes if d["modelo"] == MODELO_DETALHADO and d["atributo"] == "raca")
    linhas_sens = []
    for conjunto, reg_x, dec_x, pm_x, prox_x, n_atr in (
        ("soltura", reg_s, dec_s, resumo_pm, proxies, x_final.shape[1]),
        ("completo", reg_c, next((d for d in dec_c if d["atributo"] == "raca"), {}), resumo_pm_c, proxies_c, xc_final.shape[1]),
    ):
        linha = {"conjunto_atributos": conjunto, "n_variaveis": n_atr, "proxies_removidas": ";".join(prox_x) or "-"}
        for k in ("auc_roc", "auc_roc_ic_inferior", "auc_roc_ic_superior", "auc_pr", "brier", "acuracia",
                  "raca_disparate_impact_ratio", "raca_disparate_impact_ratio_ic_inferior",
                  "raca_disparate_impact_ratio_ic_superior", "raca_dif_fpr", "raca_dif_fpr_ic_inferior",
                  "raca_dif_fpr_ic_superior", "sexo_disparate_impact_ratio", "sexo_dif_fpr", "indice_nij",
                  "hiperparametros"):
            linha[k] = reg_x.get(k)
        linha["parcela_taxa_base_na_razao_fpr"] = dec_x.get("parcela_chances_taxa_base")
        linha["parcela_ppv_na_razao_fpr"] = dec_x.get("parcela_termo_ppv")
        linha["parcela_sensibilidade_na_razao_fpr"] = dec_x.get("parcela_termo_sensibilidade")
        linha["auc_reconstrucao_raca"] = pm_x["auc_roc_reconstrucao_raca"]
        linhas_sens.append(linha)
    pares_sens = aval_d.comparar_auc_pareado(y_te, ypb_c, probas[MODELO_DETALHADO])
    linhas_sens[1]["diferenca_auc_vs_soltura"] = pares_sens["diferenca_auc"]
    linhas_sens[1]["diferenca_auc_ic_inferior"] = pares_sens["ic_inferior"]
    linhas_sens[1]["diferenca_auc_ic_superior"] = pares_sens["ic_superior"]
    tabela_sens = pd.DataFrame(linhas_sens)
    tabela_proxy_mv = pd.DataFrame([resumo_pm, resumo_pm_c])
    registrador.publicar("sensibilidade_concluida", {"auc_soltura": reg_s["auc_roc"], "auc_completo": reg_c["auc_roc"]})

    # =====================================================================
    # LAÇO EXTERNO: repetições com sementes diferentes
    # =====================================================================
    linhas_rep = []
    for semente in sementes:
        if semente == semente_ref:
            for nome, reg in ((r["modelo"], r) for r in registros):
                linhas_rep.append({"semente": semente, **{k: reg[k] for k in (
                    "modelo", "acuracia", "auc_roc", "auc_pr", "brier", "recall", "precisao",
                    "raca_disparate_impact_ratio", "raca_dif_fpr", "raca_dif_fnr",
                    "sexo_disparate_impact_ratio", "sexo_dif_fpr", "sexo_dif_fnr",
                    "indice_nij")}, "proxies_removidas": "-" if nome == BASELINE else (";".join(proxies) or "-")})
            continue
        _log(f"Repetição do laço externo: semente {semente} ({sementes.index(semente) + 1}/{len(sementes)})")
        xs_tr0, xs_te0, ys_tr, ys_te = divisor.dividir_treino_teste(x_soltura, y, semente=semente)
        prox_s, _, rem_s = _identificar_proxies_no_treino(xs_tr0, raca.loc[xs_tr0.index])
        xs_tr, xs_te = rem_s.aplicar_remocao(xs_tr0, prox_s), rem_s.aplicar_remocao(xs_te0, prox_s)
        grupos_s = {"raca": raca.loc[xs_te.index], "sexo": sexo.loc[xs_te.index]}
        for nome in ALGORITMOS:
            modelo_s, _ = _treinar_algoritmo(nome, xs_tr, ys_tr, cv_interno)
            linha = _avaliar_rapido(nome, ys_te, modelo_s.prever(xs_te), modelo_s.prever_probabilidade(xs_te), grupos_s, aval_d, aval_e)
            linhas_rep.append({"semente": semente, **linha, "proxies_removidas": ";".join(prox_s) or "-"})
        base_s = ModeloBaselineDresselFarid()
        base_s.treinar(x_base.loc[xs_tr.index], ys_tr)
        linha = _avaliar_rapido(BASELINE, ys_te, base_s.prever(x_base.loc[xs_te.index]),
                                base_s.prever_probabilidade(x_base.loc[xs_te.index]), grupos_s, aval_d, aval_e)
        linhas_rep.append({"semente": semente, **linha, "proxies_removidas": "-"})
        registrador.publicar("repeticao_concluida", {"semente": semente})
    tabela_rep = pd.DataFrame(linhas_rep)
    tabela_resumo_rep = resumir_repeticoes(tabela_rep, "modelo", [
        "acuracia", "auc_roc", "auc_pr", "brier", "recall", "precisao",
        "raca_disparate_impact_ratio", "raca_dif_fpr", "raca_dif_fnr",
        "sexo_disparate_impact_ratio", "sexo_dif_fpr", "sexo_dif_fnr", "indice_nij"])

    # ---------------- UC10: exportar relatório ----------------
    relatorio = RelatorioResultados(formato="csv")
    resultados = relatorio.gerar(
        tabela_comparacao, tabela_ablacao,
        equidade_detalhada=tabela_equidade,
        decomposicao_disparidade_fpr=tabela_decomposicao,
        diferencas_auc_pareadas=tabela_pares,
        diagnostico_associacao_raca=diagnostico,
        diagnostico_proxy_multivariado=tabela_proxy_mv,
        importancia_proxy_multivariado=importancia_pm,
        importancia_atributos_modelo=tabela_importancia,
        pos_processamento_equidade=tabela_pos,
        analise_sensibilidade_atributos=tabela_sens,
        repeticoes_por_semente=tabela_rep,
        resumo_repeticoes=tabela_resumo_rep,
    )
    arquivos = relatorio.exportar(resultados, resultados_dir)
    duracao = round((time.time() - inicio) / 60, 1)
    # RNF05: resumo textual dos principais resultados, legível sem consultar o código
    caminho_resumo = Path(resultados_dir) / "resumo_resultados.txt"
    caminho_resumo.write_text(_resumo_textual(
        n_registros=len(y), n_teste=len(y_te), proxies=proxies, comparacao=tabela_comparacao,
        pares=tabela_pares, resumo_rep=tabela_resumo_rep, equidade=tabela_equidade,
        decomposicao=tabela_decomposicao, ablacao=tabela_ablacao, pos=tabela_pos, sens=tabela_sens,
        proxy_mv=resumo_pm, duracao=duracao), encoding="utf-8")
    arquivos.append(caminho_resumo)
    registrador.publicar("relatorio_gerado", {"arquivos": [a.name for a in arquivos], "duracao_minutos": duracao})

    # verificação de coerência: a variante final da ablação deve reproduzir o modelo principal
    final = tabela_ablacao.set_index("variante").loc["sem_raca_sem_proxies", "auc_roc"]
    if abs(final - reg_s["auc_roc"]) > 1e-9:
        print(f"AVISO: ablação (sem raça, sem proxies) AUC={final:.6f} difere do modelo principal AUC={reg_s['auc_roc']:.6f}")

    print("\n=== Comparação de modelos (divisão de referência) ===")
    print(tabela_comparacao[["modelo", "acuracia", "auc_roc", "auc_roc_ic_inferior", "auc_roc_ic_superior", "brier",
                             "raca_disparate_impact_ratio", "raca_dif_fpr", "indice_nij"]].to_string(index=False))
    print("\n=== Resumo das repetições (média) ===")
    print(tabela_resumo_rep[["modelo", "n_repeticoes", "auc_roc_media", "auc_roc_dp", "raca_dif_fpr_media"]].to_string(index=False))
    print("\n=== Sensibilidade: soltura x completo ===")
    print(tabela_sens[["conjunto_atributos", "auc_roc", "raca_dif_fpr", "auc_reconstrucao_raca"]].to_string(index=False))
    print(f"\nConcluído em {duracao} minutos. Resultados em: {resultados_dir}/")
    return tabela_comparacao


def main():
    parser = argparse.ArgumentParser(description="Pipeline de predição de reincidência (NIJ)")
    parser.add_argument("--dataset", type=str, default="data/nij_full_dataset.csv")
    parser.add_argument("--sintetico", action="store_true", help="Usa dados sintéticos (smoke test)")
    parser.add_argument("--resultados", type=str, default="results")
    parser.add_argument("--repeticoes", type=int, default=10, help="Divisões do laço externo (padrão: 10)")
    args = parser.parse_args()
    executar_pipeline(args.dataset, args.sintetico, args.resultados, args.repeticoes)


if __name__ == "__main__":
    main()
