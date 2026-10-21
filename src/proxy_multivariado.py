"""
Detecção de proxy multivariado/combinatório de raça.

A detecção de proxies por V de Cramér / correlação ponto-bisserial
(bias_removal.py) é BIVARIADA: testa cada variável isoladamente contra a
raça. É possível que nenhuma variável isolada ultrapasse o limiar e,
ainda assim, uma COMBINAÇÃO de variáveis permita reconstruir a raça.

Esta técnica treina um classificador auxiliar (Gradient Boosting) para
prever a raça a partir do conjunto de atributos finais (já sem raça e
sem os proxies bivariados) e mede sua capacidade de discriminação fora
da amostra (AUC-ROC com validação cruzada estratificada de 5 dobras; o
pré-processamento é reajustado dentro de cada dobra).

Importante para a interpretação: uma AUC alta mostra que a raça é
RECONSTRUÍVEL a partir dos atributos; isso não demonstra, por si só, que
o modelo de reincidência use esse sinal, nem que ele CAUSE a diferença
de taxas de erro entre grupos.

Régua de interpretação: equivalentes de AUC para os tamanhos de efeito
pequeno, médio e grande de Cohen (d = 0,2; 0,5; 0,8), propostos por Rice
e Harris (2005) justamente para a literatura de avaliação de risco de
reincidência: AUC de 0,556, 0,639 e 0,714.

A importância de cada variável original na reconstrução da raça é
medida por importância por permutação (queda de AUC ao embaralhar a
variável) num conjunto de validação separado (30%), com o classificador
ajustado nos outros 70%.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import Preprocessador

LIMIARES_RICE_HARRIS = (0.556, 0.639, 0.714)


def interpretar_auc(auc: float) -> str:
    """Classifica uma AUC pelos equivalentes de Rice e Harris (2005)."""
    pequeno, medio, grande = LIMIARES_RICE_HARRIS
    valor = max(auc, 1 - auc)
    if valor < pequeno:
        return "desprezível (abaixo do efeito pequeno)"
    if valor < medio:
        return "pequeno"
    if valor < grande:
        return "médio"
    return "grande"


def _pipeline_auxiliar(x: pd.DataFrame, semente: int) -> Pipeline:
    return Pipeline([
        ("prep", Preprocessador().construir_transformador(x)),
        ("clf", GradientBoostingClassifier(random_state=semente)),
    ])


def detectar_proxy_multivariado(
    x: pd.DataFrame,
    raca: pd.Series,
    grupos: tuple[str, str] | None = None,
    n_splits: int = 5,
    semente_aleatoria: int = 42,
    calcular_importancias: bool = True,
) -> tuple[dict, pd.DataFrame]:
    raca_validas = raca.dropna()
    contagens = raca_validas.value_counts()
    if grupos is None:
        if len(contagens) < 2:
            raise ValueError("São necessários ao menos dois grupos raciais para o diagnóstico de proxy multivariado.")
        grupos = tuple(contagens.index[:2])

    mascara = raca.isin(grupos)
    x_bin = x.loc[mascara]
    y_bin = (raca.loc[mascara] == grupos[0]).astype(int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semente_aleatoria)
    proba = cross_val_predict(
        _pipeline_auxiliar(x_bin, semente_aleatoria), x_bin, y_bin, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    auc = float(roc_auc_score(y_bin, proba))

    resumo = {
        "grupo_positivo": str(grupos[0]),
        "grupo_negativo": str(grupos[1]),
        "n_amostras": int(mascara.sum()),
        "n_variaveis_originais": int(x_bin.shape[1]),
        "auc_roc_reconstrucao_raca": auc,
        "tamanho_efeito_rice_harris": interpretar_auc(auc),
    }

    importancias = pd.DataFrame(columns=["variavel", "queda_auc_media", "queda_auc_dp"])
    if calcular_importancias:
        x_tr, x_va, y_tr, y_va = train_test_split(
            x_bin, y_bin, test_size=0.3, stratify=y_bin, random_state=semente_aleatoria
        )
        modelo = _pipeline_auxiliar(x_tr, semente_aleatoria).fit(x_tr, y_tr)
        perm = permutation_importance(
            modelo, x_va, y_va, scoring="roc_auc", n_repeats=5, random_state=semente_aleatoria, n_jobs=-1
        )
        importancias = pd.DataFrame({
            "variavel": x_va.columns,
            "queda_auc_media": perm.importances_mean,
            "queda_auc_dp": perm.importances_std,
        }).sort_values("queda_auc_media", ascending=False).reset_index(drop=True)
    return resumo, importancias
