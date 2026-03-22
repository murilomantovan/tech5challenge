from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st


DIRETORIO_SRC = Path(__file__).resolve().parent / "src"
if str(DIRETORIO_SRC) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_SRC))

from src.passos_magicos_dt.app_support import EXAMPLE_PROFILES, build_input_dataframe, explain_prediction
from src.passos_magicos_dt.config import MODEL_DIR
from src.passos_magicos_dt.modeling import load_model_bundle
from src.passos_magicos_dt.runtime import ensure_runtime_ready


LOGGER = logging.getLogger(__name__)

CICLOS_PROGRAMA = ["Quartzo", "Agata", "Ametista", "Topazio", "Nao informado"]
CICLOS_LABELS = {
    "Quartzo": "Quartzo",
    "Agata": "Ágata",
    "Ametista": "Ametista",
    "Topazio": "Topázio",
    "Nao informado": "Não informado",
}


def entrada_opcional_numero(rotulo: str, chave: str, valor_padrao: object = "") -> str:
    return st.text_input(rotulo, key=chave, value="" if valor_padrao is None else str(valor_padrao))


@st.cache_resource(show_spinner=False)
def preparar_runtime() -> dict[str, object]:
    return ensure_runtime_ready()


def _load_or_rebuild_model_bundle():
    preparar_runtime()
    try:
        return load_model_bundle(MODEL_DIR)
    except Exception:
        LOGGER.warning(
            "Falha ao carregar o bundle versionado do modelo; reconstruindo artefatos de runtime.",
            exc_info=True,
        )
        ensure_runtime_ready(force=True)
        return load_model_bundle(MODEL_DIR)


@st.cache_resource(show_spinner=False)
def carregar_bundle_modelo():
    return _load_or_rebuild_model_bundle()


def renderizar_grafico_explicacao(explicacao) -> plt.Figure:
    dados_grafico = explicacao.chart_data.sort_values("impact")
    cores = ["#c0392b" if valor >= 0 else "#2471a3" for valor in dados_grafico["impact"]]
    figura, eixo = plt.subplots(figsize=(8, 4.8))
    eixo.barh(dados_grafico["feature"], dados_grafico["impact"], color=cores)
    eixo.set_title(f"Principais fatores da previsão ({explicacao.method})")
    eixo.set_xlabel("Impacto relativo")
    eixo.set_ylabel("Variável transformada")
    figura.tight_layout()
    return figura


def texto_recomendacao(probabilidade: float, limiar: float) -> str:
    if probabilidade >= max(limiar, 0.7):
        return "Priorizar acompanhamento pedagógico e psicossocial nas próximas semanas."
    if probabilidade >= limiar:
        return "Acompanhar de perto engajamento e desempenho com plano preventivo."
    return "Manter acompanhamento padrão, com monitoramento recorrente dos indicadores."


def main() -> None:
    st.set_page_config(
        page_title="Passos Mágicos Analytics | Risco do próximo ciclo",
        page_icon="PM",
        layout="wide",
    )

    with st.spinner("Preparando dados e artefatos do projeto para execução..."):
        status_runtime = preparar_runtime()

    st.title("Painel preditivo de risco para o próximo ciclo")
    st.caption("Ferramenta analítica para estimar a probabilidade de risco educacional com base nos indicadores mais recentes.")

    if status_runtime.get("built"):
        st.caption("Artefatos reconstruídos a partir dos dados do repositório para garantir execução em ambiente limpo.")

    try:
        modelo, configuracao_modelo = carregar_bundle_modelo()
    except Exception as exc:
        st.error("Não foi possível carregar o bundle do modelo nem reconstruir os artefatos necessários.")
        st.exception(exc)
        st.stop()

    limiar_decisao = float(configuracao_modelo.get("selected_threshold", configuracao_modelo.get("threshold", 0.5)))

    with st.sidebar:
        st.subheader("Ferramenta")
        st.write("Passos Mágicos Analytics")
        st.caption("Simulador de risco por aluno com base nos indicadores mais recentes.")

    nome_perfil = st.selectbox("Perfil de demonstração", ["Manual"] + list(EXAMPLE_PROFILES))
    valores_padrao = EXAMPLE_PROFILES.get(nome_perfil, {})

    with st.form("formulario_predicao"):
        st.subheader("1. Perfil do aluno")
        coluna_1, coluna_2, coluna_3, coluna_4 = st.columns(4)
        with coluna_1:
            idade = entrada_opcional_numero("Idade", "idade", valores_padrao.get("idade", 12))
        with coluna_2:
            genero = st.selectbox(
                "Gênero",
                ["Masculino", "Feminino"],
                index=0 if valores_padrao.get("genero", "Masculino") == "Masculino" else 1,
            )
        with coluna_3:
            ciclo_programa = st.selectbox(
                "Ciclo do programa",
                CICLOS_PROGRAMA,
                index=CICLOS_PROGRAMA.index(valores_padrao.get("ciclo_programa", "Nao informado")),
                format_func=lambda valor: CICLOS_LABELS.get(valor, valor),
            )
        with coluna_4:
            fase_atual = st.text_input("Fase atual", value=valores_padrao.get("fase", "Fase 3"))

        st.subheader("2. Indicadores atuais")
        coluna_5, coluna_6, coluna_7, coluna_8 = st.columns(4)
        with coluna_5:
            inde_atual = entrada_opcional_numero("INDE atual", "inde_atual", valores_padrao.get("inde_atual", ""))
            inde_anterior = entrada_opcional_numero("INDE do ano anterior", "inde_anterior", valores_padrao.get("inde_anterior", ""))
            ida = entrada_opcional_numero("IDA", "ida", valores_padrao.get("ida", ""))
        with coluna_6:
            ieg = entrada_opcional_numero("IEG", "ieg", valores_padrao.get("ieg", ""))
            iaa = entrada_opcional_numero("IAA", "iaa", valores_padrao.get("iaa", ""))
            ipv = entrada_opcional_numero("IPV", "ipv", valores_padrao.get("ipv", ""))
        with coluna_7:
            ips = entrada_opcional_numero("IPS", "ips", valores_padrao.get("ips", ""))
            ipp = entrada_opcional_numero("IPP", "ipp", valores_padrao.get("ipp", ""))
            numero_avaliacoes = entrada_opcional_numero("Número de avaliações", "n_av", valores_padrao.get("n_av", ""))
        with coluna_8:
            cg = entrada_opcional_numero("CG", "cg", valores_padrao.get("cg", ""))
            cf = entrada_opcional_numero("CF", "cf", valores_padrao.get("cf", ""))
            ct = entrada_opcional_numero("CT", "ct", valores_padrao.get("ct", ""))

        with st.expander("Notas e histórico opcional"):
            coluna_9, coluna_10, coluna_11, coluna_12 = st.columns(4)
            with coluna_9:
                matematica = entrada_opcional_numero("Matemática", "mat", valores_padrao.get("mat", ""))
            with coluna_10:
                portugues = entrada_opcional_numero("Português", "por", valores_padrao.get("por", ""))
            with coluna_11:
                ingles = entrada_opcional_numero("Inglês", "ing", valores_padrao.get("ing", ""))
            with coluna_12:
                fase_ideal_numerica = entrada_opcional_numero("Fase ideal numérica", "fase_ideal_num", valores_padrao.get("fase_ideal_num", ""))
                ano_ingresso = entrada_opcional_numero("Ano de ingresso", "ano_ingresso", valores_padrao.get("ano_ingresso", ""))

        calcular_risco = st.form_submit_button("Calcular risco do próximo ciclo", type="primary")

    if not calcular_risco:
        return

    entrada_bruta = {
        "idade": idade,
        "genero": genero,
        "ciclo_programa": ciclo_programa,
        "fase": fase_atual,
        "fase_ideal_num": fase_ideal_numerica,
        "ano_ingresso": ano_ingresso,
        "inde_atual": inde_atual,
        "inde_anterior": inde_anterior,
        "cg": cg,
        "cf": cf,
        "ct": ct,
        "n_av": numero_avaliacoes,
        "iaa": iaa,
        "ieg": ieg,
        "ips": ips,
        "ipp": ipp,
        "ida": ida,
        "ipv": ipv,
        "mat": matematica,
        "por": portugues,
        "ing": ingles,
    }
    quadro_entrada = build_input_dataframe(entrada_bruta)
    probabilidade_risco = float(modelo.predict_proba(quadro_entrada)[0, 1])
    indicador_risco = int(probabilidade_risco >= limiar_decisao)

    st.subheader("Resultado")
    coluna_a, coluna_b, coluna_c = st.columns(3)
    coluna_a.metric("Probabilidade de risco", f"{probabilidade_risco * 100:.1f}%")
    coluna_b.metric("Classificação", "Alto risco" if indicador_risco else "Baixo risco")
    coluna_c.metric("Limiar usado", f"{limiar_decisao:.2f}")

    if indicador_risco:
        st.error("O aluno tem maior chance de entrar ou permanecer em risco no próximo ciclo.")
    else:
        st.success("O aluno apresenta risco controlado para o próximo ciclo, segundo o modelo.")
    st.info(texto_recomendacao(probabilidade_risco, limiar_decisao))

    st.subheader("Explicação da previsão")
    explicacao = explain_prediction(modelo, quadro_entrada)
    st.pyplot(renderizar_grafico_explicacao(explicacao), width="stretch")

    with st.expander("Dados enviados ao modelo"):
        st.dataframe(quadro_entrada, width="stretch")

if __name__ == "__main__":
    main()
