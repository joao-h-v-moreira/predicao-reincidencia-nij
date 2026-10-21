"""
Gerador de dataset SINTÉTICO com a mesma estrutura de colunas e os
mesmos FORMATOS do Full Dataset real do NIJ Recidivism Forecasting
Challenge (faixas de contagem como "3 or more", colunas booleanas,
valores ausentes, variáveis do período de supervisão, alvos por ano).

*** ATENÇÃO ***
Este módulo NÃO contém dados reais do NIJ. Ele existe exclusivamente
para o teste de integração (smoke test) do pipeline. Nenhuma métrica
calculada a partir dele deve ser interpretada como resultado do trabalho.
As relações entre variáveis foram inventadas apenas para que todas as
etapas tenham algo a detectar (um proxy geográfico, sinal preditivo no
histórico criminal e variáveis de supervisão afetadas pelo desfecho).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _faixa(valores: np.ndarray, teto: int) -> np.ndarray:
    """Converte contagens inteiras nas faixas de texto do codebook ("0", ..., "3 or more")."""
    return np.array([f"{teto} or more" if v >= teto else str(int(v)) for v in valores], dtype=object)


def _com_ausentes(valores, rng, proporcao: float):
    valores = np.array(valores, dtype=object)
    valores[rng.random(len(valores)) < proporcao] = np.nan
    return valores


def gerar_dataset_sintetico(n: int = 4000, semente: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semente)

    raca = rng.choice(["BLACK", "WHITE"], size=n, p=[0.57, 0.43])
    negro = (raca == "BLACK").astype(float)
    sexo = rng.choice(["M", "F"], size=n, p=[0.88, 0.12])

    # área de residência fortemente associada à raça (proxy a ser detectado)
    puma = np.where(rng.random(n) < 0.7,
                    np.where(negro == 1, rng.integers(1, 10, n), rng.integers(12, 26, n)),
                    rng.integers(1, 26, n))

    faixas_idade = ["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48 or older"]
    idade_idx = rng.choice(len(faixas_idade), size=n, p=[0.1, 0.17, 0.18, 0.16, 0.13, 0.11, 0.15])

    pres_fel = rng.poisson(2.5 + 0.4 * negro)
    pres_misd = rng.poisson(1.5, n)
    pres_viol = rng.poisson(0.6 + 0.2 * negro)
    pres_prop = rng.poisson(1.0, n)
    pres_drug = rng.poisson(1.0, n)
    pres_ppv = rng.poisson(0.8, n)
    cond_fel = rng.poisson(1.0 + 0.2 * negro)
    cond_misd = rng.poisson(1.0, n)
    cond_prop = rng.poisson(0.6, n)
    cond_drug = rng.poisson(0.5, n)
    rev_parole = rng.random(n) < 0.1
    rev_prob = rng.random(n) < 0.15
    gangue = rng.random(n) < (0.12 + 0.06 * negro)
    score = np.clip(np.round(3 + 0.6 * cond_fel + 0.4 * pres_fel / 2 - 0.4 * idade_idx + rng.normal(0, 1.5, n)), 1, 10)

    logit = (-0.9 + 0.30 * np.minimum(cond_fel, 3) + 0.10 * np.minimum(pres_fel, 10)
             + 0.35 * rev_parole + 0.25 * gangue - 0.18 * idade_idx + 0.10 * score
             + 0.15 * (puma < 10) + rng.normal(0, 0.4, n))
    reincide = rng.random(n) < 1 / (1 + np.exp(-logit))
    ano = rng.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])

    # variáveis do período de supervisão, em parte AFETADAS pelo desfecho
    # (quem reincide cedo tem menos dias empregado e mais violações) --
    # é o vazamento temporal que a análise principal (conjunto "soltura") evita
    cedo = reincide & (ano == 1)
    empregado = np.clip(rng.beta(2, 2, n) - 0.25 * cedo, 0, 1)
    violacao = rng.random(n) < (0.15 + 0.25 * reincide)
    teste_meth = np.clip(rng.beta(1, 15, n) + 0.03 * (1 - negro) + 0.05 * reincide, 0, 1)

    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "Gender": sexo,
        "Race": raca,
        "Age_at_Release": np.array(faixas_idade, dtype=object)[idade_idx],
        "Residence_PUMA": puma,
        "Gang_Affiliated": np.where(sexo == "F", None, gangue).astype(object),
        "Supervision_Risk_Score_First": np.where(rng.random(n) < 0.02, np.nan, score),
        "Supervision_Level_First": _com_ausentes(rng.choice(["Standard", "High", "Specialized"], n, p=[0.45, 0.35, 0.2]), rng, 0.05),
        "Education_Level": rng.choice(["Less than HS diploma", "High School Diploma", "At least some college"], n, p=[0.4, 0.4, 0.2]),
        "Dependents": _faixa(rng.poisson(1.2, n), 3),
        "Prison_Offense": _com_ausentes(rng.choice(["Violent/Sex", "Violent/Non-Sex", "Property", "Drug", "Other"], n), rng, 0.1),
        "Prison_Years": rng.choice(["Less than 1 year", "1-2 years", "Greater than 2 to 3 years", "More than 3 years"], n),
        "Prior_Arrest_Episodes_Felony": _faixa(pres_fel, 10),
        "Prior_Arrest_Episodes_Misd": _faixa(pres_misd, 6),
        "Prior_Arrest_Episodes_Violent": _faixa(pres_viol, 3),
        "Prior_Arrest_Episodes_Property": _faixa(pres_prop, 5),
        "Prior_Arrest_Episodes_Drug": _faixa(pres_drug, 5),
        "Prior_Arrest_Episodes_PPViolationCharges": _faixa(pres_ppv, 5),
        "Prior_Arrest_Episodes_DVCharges": rng.random(n) < 0.15,
        "Prior_Arrest_Episodes_GunCharges": rng.random(n) < 0.2,
        "Prior_Conviction_Episodes_Felony": _faixa(cond_fel, 3),
        "Prior_Conviction_Episodes_Misd": _faixa(cond_misd, 4),
        "Prior_Conviction_Episodes_Viol": rng.random(n) < 0.2,
        "Prior_Conviction_Episodes_Prop": _faixa(cond_prop, 3),
        "Prior_Conviction_Episodes_Drug": _faixa(cond_drug, 2),
        "Prior_Conviction_Episodes_PPViolationCharges": rng.random(n) < 0.25,
        "Prior_Conviction_Episodes_DomesticViolenceCharges": rng.random(n) < 0.08,
        "Prior_Conviction_Episodes_GunCharges": rng.random(n) < 0.1,
        "Prior_Revocations_Parole": rev_parole,
        "Prior_Revocations_Probation": rev_prob,
        "Condition_MH_SA": rng.random(n) < 0.4,
        "Condition_Cog_Ed": rng.random(n) < 0.45,
        "Condition_Other": rng.random(n) < 0.2,
        "Violations_ElectronicMonitoring": rng.random(n) < 0.1,
        "Violations_Instruction": violacao,
        "Violations_FailToReport": rng.random(n) < 0.05,
        "Violations_MoveWithoutPermission": rng.random(n) < 0.1,
        "Delinquency_Reports": _faixa(rng.poisson(0.3 + 0.5 * reincide), 4),
        "Program_Attendances": _faixa(rng.poisson(2.0, n), 10),
        "Program_UnexcusedAbsences": _faixa(rng.poisson(0.3, n), 3),
        "Residence_Changes": _faixa(rng.poisson(0.8, n), 3),
        "Avg_Days_per_DrugTest": np.where(rng.random(n) < 0.2, np.nan, rng.gamma(3, 30, n)),
        "DrugTests_THC_Positive": np.where(rng.random(n) < 0.2, np.nan, rng.beta(1, 8, n)),
        "DrugTests_Cocaine_Positive": np.where(rng.random(n) < 0.2, np.nan, rng.beta(1, 25, n)),
        "DrugTests_Meth_Positive": np.where(rng.random(n) < 0.2, np.nan, teste_meth),
        "DrugTests_Other_Positive": np.where(rng.random(n) < 0.2, np.nan, rng.beta(1, 40, n)),
        "Percent_Days_Employed": np.where(rng.random(n) < 0.02, np.nan, empregado),
        "Jobs_Per_Year": np.where(rng.random(n) < 0.03, np.nan, rng.gamma(1.2, 0.6, n)),
        "Employment_Exempt": rng.random(n) < 0.12,
        "Recidivism_Within_3years": np.where(reincide, "Yes", "No"),
        "Recidivism_Arrest_Year1": reincide & (ano == 1),
        "Recidivism_Arrest_Year2": reincide & (ano == 2),
        "Recidivism_Arrest_Year3": reincide & (ano == 3),
        "Training_Sample": (rng.random(n) < 0.7).astype(int),
    })
    return df
