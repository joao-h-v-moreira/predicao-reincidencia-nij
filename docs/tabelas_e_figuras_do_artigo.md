# Origem das tabelas e figuras do artigo

Todas as tabelas e figuras do artigo *Remover a variável racial é suficiente? Uma avaliação com dados
reais de liberdade condicional* (e do Capítulo 5 da Sexta Entrega do TCC) são geradas a partir dos
arquivos da pasta `results/` desta versão (v2.2.0), produzidos por `python run_pipeline.py` sobre o
Full Dataset do NIJ. Nenhum número foi digitado à mão: os textos são montados por scripts que leem
esses arquivos.

| Item do artigo | Conteúdo | Arquivo(s) em `results/` |
|---|---|---|
| Tabela 1 | Desempenho dos modelos (partição de referência e dez partições) | `comparacao_modelos.csv`, `resumo_repeticoes.csv`, `diferencas_auc_pareadas.csv` |
| Tabela 2 | Desenho do primeiro ano do desafio do NIJ | `comparacao_desafio_nij.csv` |
| Tabela 3 | Estudo de ablação | `estudo_ablacao.csv` |
| Tabela 4 | Comparações entre grupos (DI, diferenças de FPR, FNR, PPV e Brier skill) | `equidade_detalhada.csv` (linhas `BLACK_vs_WHITE` e `M_vs_F`) |
| Tabela 5 | Decomposição da razão de FPR (identidade de Chouldechova), com IC e dez partições | `decomposicao_disparidade_fpr.csv`, `resumo_repeticoes.csv` (colunas `*_parcela_*`) |
| Tabela 6 | Métricas por subgrupo de raça e sexo | `equidade_interseccional.csv` |
| Tabela 7 | Modelo principal e modelos sem as variáveis do sexo | `robustez_gang_affiliated.csv`, `robustez_gang_affiliated_resumo.csv`, `equidade_detalhada_variantes.csv` |
| Tabela 8 | Sensibilidade aos critérios de identificação de proxies | `sensibilidade_limiares_proxy.csv`, `sensibilidade_limiares_proxy_resumo.csv`, `sensibilidade_limiares_proxy_repeticoes.csv` |
| Tabela 9 | Ajuste de limiar por grupo racial | `pos_processamento_equidade.csv`, `pos_processamento_resumo.csv` |
| Figura 1 | Curvas de calibração por grupo racial | `calibracao_por_grupo_raca.png` |
| Figura 2 | Diferença de FPR antes e depois do ajuste de limiar nas dez partições | `pos_processamento_repeticoes.png` (dados em `pos_processamento_repeticoes.csv`) |

Números citados no texto e não tabelados vêm dos mesmos arquivos: Brier skill por grupo
(`equidade_detalhada.csv`, colunas `brier_skill` e `dif_brier_skill`; dez partições em
`resumo_repeticoes.csv`, colunas `raca_dif_brier_skill_*`), importância por permutação
(`importancia_atributos_modelo.csv`), diagnóstico de proxies (`diagnostico_associacao_raca.csv`,
`diagnostico_proxy_multivariado.csv`, `importancia_proxy_multivariado.csv`), análise de
sensibilidade (`analise_sensibilidade_atributos.csv`), taxas-base (`taxas_base_reincidencia.csv`) e
duração da execução (`eventos.jsonl`). O modelo das análises detalhadas é definido pela constante
`MODELO_DETALHADO` de `run_pipeline.py`.
