"""
Classe Dataset (Camada de Ingestão).

Responsável por importar o Full Dataset do NIJ Recidivism Forecasting
Challenge (UC01) a partir de um arquivo CSV local. Suporta importação
completa ou restrita a um subconjunto de colunas (FA01), já que o Full
Dataset é fornecido em uma única tabela (não há múltiplas tabelas a
selecionar, ao contrário do que constava, por engano, em uma versão
anterior deste documento, que mencionava as tabelas do COMPAS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

COLUNA_ALVO_PADRAO = "Recidivism_Within_3years"
COLUNA_RACA_PADRAO = "Race"

# Colunas que NÃO podem entrar como atributo preditivo, por vazarem
# diretamente a variável-alvo. Recidivism_Arrest_Year1/2/3 compõem, por
# construção, o próprio Recidivism_Within_3years (reincidência no ano 1,
# 2 ou 3) -- se entrarem como atributos, o modelo "prevê" o alvo lendo a
# resposta já embutida nos dados, em vez de aprender um padrão real
# (data leakage). ID é apenas um identificador de linha, sem valor
# preditivo, e também é excluído por boa prática.
COLUNAS_EXCLUIDAS_DOS_ATRIBUTOS = [
    "Recidivism_Arrest_Year1",
    "Recidivism_Arrest_Year2",
    "Recidivism_Arrest_Year3",
    "ID",
    # "Training_Sample" não pertence ao codebook oficial (Apêndice 2) do
    # NIJ Recidivism Forecasting Challenge -- é um artefato administrativo
    # do próprio desafio (indicador de qual lote treino/teste o NIJ usou
    # na competição original), não um atributo real da pessoa. Associação
    # com raça é desprezível (V ~ 0,008 em testes empíricos), então não é
    # um proxy; é excluída por não ser uma informação válida sobre o
    # indivíduo para fins de predição.
    # Ela é usada só para reproduzir a divisão oficial treino/teste do
    # desafio na comparação com os resultados do NIJ
    # (analises_complementares.comparar_com_desafio_nij).
    "Training_Sample",
]

# Variáveis medidas DURANTE a liberdade condicional, isto é, depois do
# momento da soltura e dentro do mesmo período de 3 anos em que a
# reincidência é contada. O relatório oficial do NIJ registra que essas
# variáveis "não foram fornecidas para o primeiro período" do desafio,
# para espelhar a situação real em que, no momento da soltura, essa
# informação ainda não existe (WHITE et al., 2022). Usá-las para prever a
# reincidência em 3 anos contados da soltura introduz vazamento temporal:
# para quem é preso de novo cedo, parte desses valores é consequência da
# própria nova prisão (ex.: Percent_Days_Employed cai porque a pessoa foi
# presa), e não informação disponível antes dela.
#
# Por isso, a análise principal do trabalho (escolha "A") usa apenas o
# conjunto "soltura" (tudo, menos estas colunas); o conjunto "completo"
# (com estas colunas) é usado só como análise de sensibilidade, para
# medir quanto do desempenho e da disparidade vinha dessa informação
# posterior à soltura.
COLUNAS_PERIODO_SUPERVISAO = [
    "Violations_ElectronicMonitoring",
    "Violations_Instruction",
    "Violations_FailToReport",
    "Violations_MoveWithoutPermission",
    "Delinquency_Reports",
    "Program_Attendances",
    "Program_UnexcusedAbsences",
    "Residence_Changes",
    "Avg_Days_per_DrugTest",
    "DrugTests_THC_Positive",
    "DrugTests_Cocaine_Positive",
    "DrugTests_Meth_Positive",
    "DrugTests_Other_Positive",
    "Percent_Days_Employed",
    "Jobs_Per_Year",
    "Employment_Exempt",
]

# "soltura": somente o que já se sabe quando a pessoa é solta (histórico
# criminal, dados da pena cumprida, dados demográficos, score e nível de
# supervisão atribuídos no primeiro contato, condições da condicional).
# "completo": soltura + variáveis do período de supervisão.
CONJUNTOS_ATRIBUTOS = ("soltura", "completo")


def selecionar_conjunto_atributos(df: pd.DataFrame, conjunto: str) -> pd.DataFrame:
    """Retorna `df` restrito ao conjunto de atributos pedido."""
    if conjunto not in CONJUNTOS_ATRIBUTOS:
        raise ValueError(f"Conjunto de atributos desconhecido: {conjunto}. Use um de {CONJUNTOS_ATRIBUTOS}.")
    if conjunto == "completo":
        return df
    return df.drop(columns=[c for c in COLUNAS_PERIODO_SUPERVISAO if c in df.columns])


# Algumas colunas do Full Dataset chegam ao CSV sem cabeçalho legível
# (possivelmente por um problema de codificação/exportação a montante,
# anterior à distribuição do NIJ) e o pandas as nomeia "_v1".."_v4". A
# posição dessas colunas no arquivo bate exatamente com 4 variáveis do
# codebook oficial que, por coincidência, ficam "faltando" nos nomes
# visíveis:
#   - uma entre Prior_Arrest_Episodes_Drug e Prior_Arrest_Episodes_DVCharges
#     (o grupo Prior_Arrest_Episodes_* tem 8 subtipos no codebook, mas só
#     7 aparecem nomeados no CSV real)
#   - três entre Prior_Conviction_Episodes_Drug e Prior_Revocations_Parole
#     (idem: o codebook lista Prior_Conviction_Episodes_PPViolationCharges,
#     _DomesticViolenceCharges e _GunCharges nas posições 26-28, logo após
#     Prior_Conviction_Episodes_Drug na posição 25)
# Renomeamos por posição/nome de origem para que essas variáveis entrem
# no pipeline com seus nomes reais, em vez de ficarem opacas em todos os
# relatórios (diagnóstico de associação, importâncias, etc.).
MAPA_RENOMEACAO_COLUNAS_SEM_CABECALHO = {
    "_v1": "Prior_Arrest_Episodes_PPViolationCharges",
    "_v2": "Prior_Conviction_Episodes_PPViolationCharges",
    "_v3": "Prior_Conviction_Episodes_DomesticViolenceCharges",
    "_v4": "Prior_Conviction_Episodes_GunCharges",
}

# Colunas que o pandas lê como numéricas (por serem códigos) mas que são,
# na verdade, categóricas nominais -- sem ordem ou magnitude real entre
# os valores. Residence_PUMA é um código de área geográfica (Public Use
# Microdata Area); tratá-la como número contínuo (i) alimentaria o
# modelo com uma "escala" que não existe e (ii) faria o RemovedorDeVies
# testá-la com correlação ponto-bisserial em vez de V de Cramér,
# mascarando um proxy clássico de raça na literatura de justiça
# criminal (associação entre área de residência e raça, efeito
# "redlining").
COLUNAS_CATEGORICAS_CODIFICADAS_NUMERICAMENTE = [
    "Residence_PUMA",
]


class Dataset:
    def __init__(
        self,
        caminho_arquivo: str | Path,
        coluna_alvo: str = COLUNA_ALVO_PADRAO,
        coluna_raca: str = COLUNA_RACA_PADRAO,
    ):
        self.caminho_arquivo = Path(caminho_arquivo)
        self.coluna_alvo = coluna_alvo
        self.coluna_raca = coluna_raca
        self.dataframe: Optional[pd.DataFrame] = None

    def importar(self, colunas: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Importa o dataset. Se `colunas` for informado (FA01), apenas o
        subconjunto de colunas solicitado é carregado, sempre incluindo a
        coluna-alvo e a coluna de raça, mesmo que não tenham sido
        explicitamente listadas (a raça é necessária à jusante para a
        avaliação de equidade, ainda que nunca entre como atributo
        preditivo).
        """
        if not self.caminho_arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.caminho_arquivo}. "
                "Baixe o Full Dataset do NIJ (ver README) antes de executar "
                "o pipeline com dados reais, ou utilize "
                "gerar_dataset_sintetico() para um teste de integração."
            )

        usecols = None
        if colunas is not None:
            colunas = set(colunas) | {self.coluna_alvo, self.coluna_raca}
            usecols = lambda c: c in colunas  # noqa: E731

        self.dataframe = pd.read_csv(self.caminho_arquivo, usecols=usecols)

        # renomeia colunas que chegaram sem cabeçalho legível (ver
        # MAPA_RENOMEACAO_COLUNAS_SEM_CABECALHO acima) para seus nomes
        # reais do codebook oficial, só quando o nome de origem (_v1 etc.)
        # de fato estiver presente -- não falha silenciosamente se o CSV
        # já vier com os nomes corretos ou com um layout diferente
        colunas_a_renomear = {
            origem: destino
            for origem, destino in MAPA_RENOMEACAO_COLUNAS_SEM_CABECALHO.items()
            if origem in self.dataframe.columns
        }
        if colunas_a_renomear:
            self.dataframe = self.dataframe.rename(columns=colunas_a_renomear)

        # força a leitura de colunas categóricas codificadas como número
        # (ex.: Residence_PUMA) como texto, para que sejam tratadas como
        # categóricas em todas as etapas seguintes (EDA, codificação,
        # identificação de proxies)
        for col in COLUNAS_CATEGORICAS_CODIFICADAS_NUMERICAMENTE:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].astype("string")

        if self.coluna_alvo not in self.dataframe.columns:
            raise ValueError(
                f"Coluna-alvo '{self.coluna_alvo}' não encontrada no "
                "arquivo. Verifique se o CSV corresponde ao Full Dataset "
                "do NIJ (ver README)."
            )
        return self.dataframe

    # também acessível como método estático da classe (diagrama de classes, Figura 2)
    selecionar_conjunto_atributos = staticmethod(selecionar_conjunto_atributos)

    def resumo(self) -> dict:
        if self.dataframe is None:
            raise RuntimeError("Chame importar() antes de resumo().")
        return {
            "n_linhas": int(self.dataframe.shape[0]),
            "n_colunas": int(self.dataframe.shape[1]),
            "colunas": list(self.dataframe.columns),
            "percentual_reincidencia": float(
                self.dataframe[self.coluna_alvo].mean()
            ),
        }
