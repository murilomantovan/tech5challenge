from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


DIRETORIO_RAIZ = Path(__file__).resolve().parents[1]
DIRETORIO_SRC = DIRETORIO_RAIZ / "src"
if str(DIRETORIO_SRC) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_SRC))

from src.passos_magicos_dt.painel_analitico_app import (
    load_feature_importance,
    load_painel_payload,
    load_storyboard,
)
from src.passos_magicos_dt.runtime import ensure_runtime_ready


def resolver_caminho_ativo(caminho_texto: str) -> Path:
    caminho = Path(caminho_texto)
    return caminho if caminho.is_absolute() else DIRETORIO_RAIZ / caminho


def renderizar_cartao_metrica(rotulo: str, valor: str) -> None:
    st.markdown(
        f"""
        <div class="pm-metric-card">
            <div class="pm-metric-label">{rotulo}</div>
            <div class="pm-metric-value">{valor}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def renderizar_visual_modelagem() -> None:
    st.markdown(
        """
        <div class="pm-model-card">
            <div class="pm-model-title">Janela temporal da modelagem</div>
            <div class="pm-model-line">
                <div class="pm-model-node">
                    <div class="pm-model-dot"></div>
                    <div class="pm-model-year">2022</div>
                    <div class="pm-model-step">Seleção</div>
                </div>
                <div class="pm-model-node">
                    <div class="pm-model-dot"></div>
                    <div class="pm-model-year">2023</div>
                    <div class="pm-model-step">Holdout</div>
                </div>
                <div class="pm-model-node">
                    <div class="pm-model-dot"></div>
                    <div class="pm-model-year">2024</div>
                    <div class="pm-model-step">Teste final</div>
                </div>
            </div>
            <div class="pm-model-notes">
                <div class="pm-model-note">Alvo: risco_próximo_ano</div>
                <div class="pm-model-note">Sem usar IAN atual</div>
                <div class="pm-model-note">Threshold guiado por F2</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def renderizar_topico(bloco: dict[str, object]) -> None:
    st.markdown(
        f"""
        <div class="pm-story-head">
            <div>
                <div class="pm-story-kicker">Tópico {bloco['scene_number']} | {bloco['label']}</div>
                <div class="pm-story-title">{bloco['title']}</div>
            </div>
            <div class="pm-story-metric-wrap">
                <div class="pm-story-metric">{bloco['highlight_metric']}</div>
                <div class="pm-story-caption">{bloco['highlight_caption']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    texto_coluna, visual_coluna = st.columns([0.34, 0.66], gap="large")

    with texto_coluna:
        st.markdown(
            f"""
            <div class="pm-takeaway-card">
                <div class="pm-takeaway-text">{bloco['screen_takeaway']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for texto in bloco["on_screen_text"]:
            st.markdown(
                f"""
                <div class="pm-bullet-line">{texto}</div>
                """,
                unsafe_allow_html=True,
            )

    with visual_coluna:
        caminho_figura = resolver_caminho_ativo(str(bloco["asset_path"]))
        if caminho_figura.exists() and caminho_figura.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            st.image(str(caminho_figura), width="stretch")
        elif bloco.get("scene_id") == "05_modelagem":
            renderizar_visual_modelagem()
        else:
            st.info(f"Visual de referência: {caminho_figura}")

    st.divider()


@st.cache_resource(show_spinner=False)
def preparar_runtime() -> dict[str, object]:
    return ensure_runtime_ready()


def main() -> None:
    st.set_page_config(
        page_title="Passos Mágicos Analytics | Painel Analítico",
        page_icon="PM",
        layout="wide",
    )

    with st.spinner("Preparando dados e artefatos analíticos para execução..."):
        status_runtime = preparar_runtime()

    carga_painel = load_painel_payload()
    if carga_painel is None:
        st.error("Payload analítico não encontrado. A reconstrução automática do ambiente não conseguiu gerar os artefatos necessários.")
        st.stop()

    resumo_painel = carga_painel["resumo_painel"]
    importancia_variaveis = load_feature_importance()
    storyboard = load_storyboard() or []
    topicos_analise = [bloco for bloco in storyboard if bloco.get("scene_id") != "08_app_fechamento"]

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f8f5ef 0%, #f4f1ea 100%);
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.8rem;
                max-width: 1380px;
            }
            h1, h2, h3 {
                color: #1d332e;
            }
            .pm-metric-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid #ddd5c9;
                border-radius: 18px;
                padding: 18px 20px;
                min-height: 108px;
            }
            .pm-metric-label {
                font-size: 0.78rem;
                color: #7b6e5e;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .pm-metric-value {
                font-size: 1.8rem;
                font-weight: 800;
                color: #1d332e;
                margin-top: 8px;
            }
            .pm-story-head {
                display: flex;
                justify-content: space-between;
                gap: 24px;
                align-items: flex-end;
                margin-top: 0.4rem;
                margin-bottom: 0.9rem;
            }
            .pm-story-kicker {
                color: #8a7558;
                font-size: 0.76rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
            }
            .pm-story-title {
                color: #1d332e;
                font-size: 1.9rem;
                font-weight: 800;
                line-height: 1.1;
                margin-top: 8px;
            }
            .pm-story-metric-wrap {
                text-align: right;
                min-width: 180px;
            }
            .pm-story-metric {
                color: #1f4b42;
                font-size: 1.8rem;
                font-weight: 800;
                line-height: 1;
            }
            .pm-story-caption {
                color: #8a7b68;
                font-size: 0.9rem;
                margin-top: 6px;
            }
            .pm-takeaway-card {
                background: #efe3d2;
                border-left: 4px solid #c48d43;
                border-radius: 14px;
                padding: 14px 16px;
                margin-bottom: 0.9rem;
            }
            .pm-takeaway-text {
                color: #433328;
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.45;
            }
            .pm-bullet-line {
                color: #2f3f3b;
                font-size: 0.96rem;
                line-height: 1.5;
                padding: 0.72rem 0;
                border-bottom: 1px solid #ded7cb;
            }
            .pm-model-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid #ddd5c9;
                border-radius: 24px;
                padding: 24px 26px 28px;
            }
            .pm-model-title {
                color: #1d332e;
                font-size: 1.08rem;
                font-weight: 700;
                margin-bottom: 1.6rem;
            }
            .pm-model-line {
                position: relative;
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 20px;
                margin-bottom: 1.7rem;
            }
            .pm-model-line::before {
                content: "";
                position: absolute;
                left: 11%;
                right: 11%;
                top: 22px;
                height: 6px;
                background: #1d332e;
                border-radius: 999px;
            }
            .pm-model-node {
                position: relative;
                text-align: center;
                z-index: 1;
            }
            .pm-model-dot {
                width: 34px;
                height: 34px;
                margin: 0 auto 0.8rem;
                border-radius: 999px;
                background: #cb9440;
                border: 4px solid #1d332e;
            }
            .pm-model-year {
                color: #1d332e;
                font-size: 1.05rem;
                font-weight: 800;
            }
            .pm-model-step {
                color: #8a7558;
                font-size: 0.92rem;
                margin-top: 0.25rem;
            }
            .pm-model-notes {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
            }
            .pm-model-note {
                background: #f5f0e7;
                border: 1px solid #ddd5c9;
                border-radius: 16px;
                color: #2f3f3b;
                font-size: 0.95rem;
                padding: 14px 16px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Painel Analítico")
    st.caption("Leitura gerencial dos resultados analíticos e do modelo preditivo, com a narrativa do vídeo organizada de forma mais direta.")

    if status_runtime.get("built"):
        st.caption("Os artefatos analíticos foram reconstruídos automaticamente a partir dos dados versionados no repositório.")

    coluna_principal, coluna_metricas = st.columns([1.15, 1], gap="large")
    with coluna_principal:
        st.subheader("Síntese gerencial")
        st.write(
            "A base do PEDE mostra melhora consistente no IAN e redução do risco atual, "
            "mas ainda há uma parcela relevante de alunos vulneráveis. "
            "O painel abaixo organiza essa história com foco maior no visual e com o texto escrito como apoio."
        )
    with coluna_metricas:
        colunas_superiores = st.columns(2)
        with colunas_superiores[0]:
            renderizar_cartao_metrica("Ano mais recente", resumo_painel["ano_mais_recente"])
        with colunas_superiores[1]:
            renderizar_cartao_metrica("Risco atual", resumo_painel["taxa_risco_mais_recente"])
        colunas_inferiores = st.columns(2)
        with colunas_inferiores[0]:
            renderizar_cartao_metrica("Modelo final", resumo_painel["modelo_final"])
        with colunas_inferiores[1]:
            renderizar_cartao_metrica("F2 holdout", resumo_painel["f2_holdout"])

    abas = st.tabs(["Resumo", "Análise", "Modelo"])

    with abas[0]:
        colunas_topo = st.columns(4)
        metricas_resumo = [
            ("Base analítica", resumo_painel["base_registros"]),
            ("Alunos únicos", resumo_painel["alunos_unicos"]),
            ("Pares de modelagem", resumo_painel["pares_modelagem"]),
            ("IAN médio 2024", resumo_painel["ian_medio_mais_recente"]),
        ]
        for coluna, (rotulo, valor) in zip(colunas_topo, metricas_resumo):
            with coluna:
                renderizar_cartao_metrica(rotulo, valor)

        quadro_resumo = pd.DataFrame(
            [
                ("Variação IDA 2022->2024", resumo_painel["variacao_ida_2022_2024"]),
                ("Recall holdout", resumo_painel["recall_holdout"]),
                ("Precisão holdout", resumo_painel["precision_holdout"]),
                ("Threshold final", resumo_painel["threshold_final"]),
            ],
            columns=["Indicador", "Valor"],
        )
        st.dataframe(quadro_resumo, width="stretch", hide_index=True)

    with abas[1]:
        if topicos_analise:
            for bloco in topicos_analise:
                renderizar_topico(bloco)
        else:
            st.warning("Storyboard da análise não encontrado. Verifique se o arquivo de runtime foi gerado corretamente.")

    with abas[2]:
        st.subheader("Leitura simples do modelo")
        st.write(
            "O modelo foi treinado para prever quem pode entrar ou permanecer em risco no próximo ciclo. "
            "A escolha final preserva rigor temporal: seleção em `2022 -> 2023` e holdout em `2023 -> 2024`."
        )

        colunas_modelo = st.columns(3)
        with colunas_modelo[0]:
            renderizar_cartao_metrica("Recall", resumo_painel["recall_holdout"])
        with colunas_modelo[1]:
            renderizar_cartao_metrica("Precisão", resumo_painel["precision_holdout"])
        with colunas_modelo[2]:
            renderizar_cartao_metrica("Threshold", resumo_painel["threshold_final"])

        caminho_matriz = resolver_caminho_ativo("artifacts/analytics/09_modelo_holdout.png")
        if caminho_matriz.exists():
            st.image(
                str(caminho_matriz),
                width="stretch",
                caption="Comparação com baseline e matriz de confusão do holdout",
            )

        if importancia_variaveis is not None and not importancia_variaveis.empty:
            principais_variaveis = importancia_variaveis.head(12).set_index("feature")
            st.subheader("Principais variáveis do modelo")
            st.bar_chart(principais_variaveis["importance"], width="stretch")

        st.markdown(
            "**Interpretação operacional:** o modelo não substitui a equipe, mas organiza prioridades e antecipa quem precisa de ação preventiva."
        )


if __name__ == "__main__":
    main()
