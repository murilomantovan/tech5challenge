# Passos Mágicos Analytics

Projeto da fase 5 do Datathon FIAP com foco em análise educacional e previsão de risco para o ciclo seguinte.

O repositório cobre três frentes:

- preparação e padronização da base do PEDE;
- treinamento do modelo preditivo de `risco_proximo_ano`;
- disponibilização dos resultados em Streamlit.

## Visão geral

O pipeline parte das bases anuais de 2022, 2023 e 2024, monta uma base analítica por aluno, gera pares temporais para modelagem e treina um classificador para estimar quem pode entrar ou permanecer em risco no ano seguinte.

O app em Streamlit tem duas páginas:

- `app.py`: simulação individual de risco;
- `pages/2_Painel_Analitico.py`: leitura analítica e resumo gerencial.

## Estrutura

```text
.
├── app.py
├── pages/
├── src/passos_magicos_dt/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── analytics/
│   ├── logs/
│   └── model/
├── notebooks/
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Como rodar localmente

```bash
pip install -r requirements.txt
python -m passos_magicos_dt build-all
streamlit run app.py
```

## Notebooks

- `notebooks/01_tratamento_e_bases_modelo.ipynb`: gera bases e artefatos.
- `notebooks/02_painel_analitico.ipynb`: revisa os resultados analíticos.
- `notebooks/03_executar_entrega_e_streamlit.ipynb`: roda o fluxo local e sobe o app.

## O que precisa subir para o GitHub

Obrigatório para o projeto funcionar:

- `app.py`
- `pages/`
- `src/`
- `data/raw/`
- `requirements.txt`
- `pyproject.toml`
- `.streamlit/` se você usar configuração própria do Streamlit

Opcional, mas útil para evitar rebuild no primeiro acesso:

- `artifacts/model/`
- `artifacts/analytics/`
- `data/processed/`

## O que pode ficar fora

Estas pastas não são necessárias para o app funcionar no Streamlit:

- `support/`
- `references/`
- `archive/legacy/`
- `artifacts/logs/`

Se os artefatos de modelo e analytics não forem publicados, o runtime tenta reconstruir tudo a partir de `data/raw/`.

## Testes

```bash
python -m unittest tests.test_pipeline
```

## Documentação

- [DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md)
