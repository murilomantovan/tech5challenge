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

## Documentação

- [DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md)
