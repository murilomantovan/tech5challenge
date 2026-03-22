# Documentação Técnica

## Escopo

O projeto lê as bases anuais do PEDE, monta a base de trabalho por aluno, treina um modelo para prever `risco_proximo_ano` e publica os resultados no Streamlit.

## Entrada de dados

Arquivo principal:

- `data/raw/PEDE_PASSOS_MAGICOS_FIAP.xlsx`

Arquivos auxiliares:

- `archive/legacy/data/` pode conter um CSV antigo, mas não é obrigatório para o app.

Leitura e preparação:

- `src/passos_magicos_dt/data.py`

## Saídas geradas pelo pipeline

Bases:

- `data/processed/base_analitica.parquet`
- `data/processed/base_modelagem_proximo_ano.parquet`

Modelo:

- `artifacts/model/model_pipeline.joblib`
- `artifacts/model/model_config.json`
- `artifacts/model/metrics_holdout.csv`
- `artifacts/model/feature_importance.csv`

Painel analítico:

- `artifacts/analytics/painel_analitico.json`
- `artifacts/analytics/resumo_analitico.md`
- figuras em `artifacts/analytics/`

## Pipeline

### 1. Preparação dos dados

Responsável:

- `src/passos_magicos_dt/data.py`

Passos principais:

1. localizar o Excel oficial;
2. ler as abas `PEDE2022`, `PEDE2023` e `PEDE2024`;
3. padronizar colunas e tipos;
4. calcular indicadores derivados;
5. gerar a base analítica e a base temporal de pares.

Transformações centrais:

- `risco_atual = 1` quando `ian <= 5`;
- `delta_inde = inde_atual - inde_anterior`;
- `media_academica = média de mat, por, ing`;
- `media_comportamental = média de iaa, ieg, ips, ipp`;
- `anos_no_programa = ano_referencia - ano_ingresso`.

### 2. Modelagem

Responsável:

- `src/passos_magicos_dt/modeling.py`

Janela temporal:

- seleção: `2022 -> 2023`
- holdout final: `2023 -> 2024`

Modelos avaliados:

- regressão logística
- random forest
- xgboost, quando disponível
- baseline de regra de negócio

Critério de escolha:

- o threshold é escolhido com foco em `F2`, respeitando precisão mínima;
- o modelo final salvo hoje é `regressao_logistica`.

### 3. Análises e figuras

Responsável:

- `src/passos_magicos_dt/analysis.py`

O módulo gera:

- respostas analíticas;
- análises complementares;
- figuras em `artifacts/analytics/`;
- resumo usado pela página analítica.

### 4. Materiais de execução

Responsável:

- `src/passos_magicos_dt/materials.py`

Esse módulo gera:

- `notebooks/01_tratamento_e_bases_modelo.ipynb`
- `notebooks/02_painel_analitico.ipynb`
- `notebooks/03_executar_entrega_e_streamlit.ipynb`

### 5. Interface

Arquivos principais:

- `app.py`
- `pages/2_Painel_Analitico.py`

O app principal faz a previsão individual. A segunda página lê os artefatos analíticos já gerados.

## Execução

Pipeline completo:

```bash
python -m passos_magicos_dt build-all
```

App:

```bash
streamlit run app.py
```

Testes:

```bash
python -m unittest tests.test_pipeline
```