from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ANALYTICS_DIR, ROOT_DIR
from .modeling import TrainingArtifacts


plt.style.use("ggplot")


@dataclass
class AnalyticsArtifacts:
    figure_paths: dict[str, Path]
    qna_items: list[dict[str, str]]
    extra_items: list[dict[str, str]]
    resumo_painel: dict[str, str]


def save_figure(fig: plt.Figure, filename: str) -> Path:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYTICS_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}"


def format_month_year(value: object) -> str:
    return f"01/{int(value)}"


def apply_month_year_index(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    data = data.copy()
    data.index = [format_month_year(value) for value in data.index]
    return data


def add_trend_line(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series, color: str = "#203040") -> None:
    valid = x_values.notna() & y_values.notna()
    if valid.sum() < 3:
        return
    slope, intercept = np.polyfit(x_values[valid], y_values[valid], deg=1)
    xs = np.linspace(float(x_values[valid].min()), float(x_values[valid].max()), 50)
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=2, linestyle="--")


def build_item(
    *,
    item_id: str,
    title: str,
    answer: str,
    key_number: str,
    figure: Path,
    managerial_takeaway: str,
    recommended_action: str,
) -> dict[str, str]:
    return {
        "id": item_id,
        "title": title,
        "answer": answer,
        "key_number": key_number,
        "figure": portable_path(figure),
        "managerial_takeaway": managerial_takeaway,
        "recommended_action": recommended_action,
    }


def build_q1(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    counts = (
        base.groupby(["ano_referencia", "categoria_ian"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"Nao informado": "Não informado"})
        .reindex(columns=["Defasagem severa", "Defasagem moderada", "Adequado", "Não informado"], fill_value=0)
    )
    risk_rate = base.groupby("ano_referencia")["risco_atual"].mean()
    counts_plot = apply_month_year_index(counts)
    risk_rate_plot = apply_month_year_index(risk_rate)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts_plot.plot(kind="bar", stacked=True, ax=axes[0], color=["#c0392b", "#f39c12", "#2e86de", "#95a5a6"])
    axes[0].set_title("Perfil de IAN por ano")
    axes[0].set_xlabel("Mês/ano de referência")
    axes[0].set_ylabel("Quantidade de alunos")
    risk_rate_plot.plot(marker="o", linewidth=2.5, ax=axes[1], color="#c0392b")
    axes[1].set_title("Taxa de risco atual ao longo dos anos")
    axes[1].set_xlabel("Mês/ano de referência")
    axes[1].set_ylabel("Taxa de risco")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
    path = save_figure(fig, "01_ian_perfil_evolucao.png")
    latest_year = int(risk_rate.index.max())
    answer = (
        f"A taxa de alunos em risco caiu de {format_pct(float(risk_rate.loc[2022]))} em 2022 "
        f"para {format_pct(float(risk_rate.loc[latest_year]))} em {latest_year}."
    )
    return path, build_item(
        item_id="1",
        title="Adequacao do nivel (IAN)",
        answer=answer,
        key_number=format_pct(float(risk_rate.loc[latest_year])),
        figure=path,
        managerial_takeaway="O programa melhora a adequacao de nivel ao longo do tempo, mas ainda termina 2024 com uma base relevante em risco.",
        recommended_action="Manter trilhas preventivas para alunos em defasagem moderada antes que migrem para risco severo.",
    )


def build_q2(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    ida_year = base.groupby("ano_referencia")["ida"].mean()
    ida_cycle = base.pivot_table(index="ano_referencia", columns="ciclo_programa", values="ida", aggfunc="mean")
    ida_year_plot = apply_month_year_index(ida_year)
    ida_cycle_plot = apply_month_year_index(ida_cycle)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ida_year_plot.plot(marker="o", linewidth=2.5, ax=axes[0], color="#1f618d")
    axes[0].set_title("IDA médio por ano")
    axes[0].set_xlabel("Mês/ano de referência")
    axes[0].set_ylabel("IDA médio")
    ida_cycle_plot.plot(marker="o", linewidth=2.0, ax=axes[1])
    axes[1].set_title("IDA médio por ciclo")
    axes[1].set_xlabel("Mês/ano de referência")
    axes[1].set_ylabel("IDA médio")
    axes[1].legend(title="Ciclo", fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
    path = save_figure(fig, "02_ida_evolucao.png")
    answer = (
        f"O IDA medio saiu de {ida_year.min():.2f} para {ida_year.max():.2f} no periodo, "
        f"com leve acomodacao em 2024 frente a 2023."
    )
    return path, build_item(
        item_id="2",
        title="Desempenho academico (IDA)",
        answer=answer,
        key_number=format_delta(float(ida_year.loc[2024] - ida_year.loc[2022])),
        figure=path,
        managerial_takeaway="O ganho academico existe, mas nao e linear em todos os ciclos.",
        recommended_action="Direcionar reforco academico aos ciclos em que o IDA desacelera para evitar acomodacao do aprendizado.",
    )


def build_q3(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    corr_ieg_ida = float(base["ieg"].corr(base["ida"]))
    corr_ieg_ipv = float(base["ieg"].corr(base["ipv"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(base["ieg"], base["ida"], alpha=0.25, color="#117864")
    add_trend_line(axes[0], base["ieg"], base["ida"])
    axes[0].set_title(f"IEG x IDA (corr={corr_ieg_ida:.2f})")
    axes[0].set_xlabel("IEG")
    axes[0].set_ylabel("IDA")
    axes[1].scatter(base["ieg"], base["ipv"], alpha=0.25, color="#884ea0")
    add_trend_line(axes[1], base["ieg"], base["ipv"])
    axes[1].set_title(f"IEG x IPV (corr={corr_ieg_ipv:.2f})")
    axes[1].set_xlabel("IEG")
    axes[1].set_ylabel("IPV")
    path = save_figure(fig, "03_ieg_vs_ida_ipv.png")
    answer = (
        f"O engajamento tem relacao positiva com o desempenho e o ponto de virada: "
        f"corr(IEG, IDA)={corr_ieg_ida:.2f} e corr(IEG, IPV)={corr_ieg_ipv:.2f}."
    )
    return path, build_item(
        item_id="3",
        title="Engajamento nas atividades (IEG)",
        answer=answer,
        key_number=f"{corr_ieg_ida:.2f} / {corr_ieg_ipv:.2f}",
        figure=path,
        managerial_takeaway="Engajamento nao e so participacao: ele antecipa resultado academico e virada positiva.",
        recommended_action="Usar quedas de IEG como gatilho operacional para intervencao rapida por monitoria e relacionamento.",
    )


def build_q4(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    corr_iaa_ida = float(base["iaa"].corr(base["ida"]))
    corr_iaa_ieg = float(base["iaa"].corr(base["ieg"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(base["iaa"], base["ida"], alpha=0.25, color="#d35400")
    add_trend_line(axes[0], base["iaa"], base["ida"])
    axes[0].set_title(f"IAA x IDA (corr={corr_iaa_ida:.2f})")
    axes[0].set_xlabel("IAA")
    axes[0].set_ylabel("IDA")
    axes[1].scatter(base["iaa"], base["ieg"], alpha=0.25, color="#5b2c6f")
    add_trend_line(axes[1], base["iaa"], base["ieg"])
    axes[1].set_title(f"IAA x IEG (corr={corr_iaa_ieg:.2f})")
    axes[1].set_xlabel("IAA")
    axes[1].set_ylabel("IEG")
    path = save_figure(fig, "04_iaa_coerencia.png")
    answer = (
        f"A autoavaliacao e parcialmente coerente com o desempenho real, "
        f"mas a correlacao e fraca: corr(IAA, IDA)={corr_iaa_ida:.2f} e corr(IAA, IEG)={corr_iaa_ieg:.2f}."
    )
    return path, build_item(
        item_id="4",
        title="Autoavaliacao (IAA)",
        answer=answer,
        key_number=f"{corr_iaa_ida:.2f}",
        figure=path,
        managerial_takeaway="A percepcao do aluno ajuda, mas sozinha nao explica o que realmente acontece no desempenho.",
        recommended_action="Combinar autoavaliacao com sinais objetivos de engajamento e desempenho antes de decidir uma intervencao.",
    )


def build_q5(base_pares: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    grouped = base_pares.groupby("risco_proximo_ano")["ips"].agg(["mean", "median", "count"])
    fig, ax = plt.subplots(figsize=(8, 5))
    base_pares.boxplot(column="ips", by="risco_proximo_ano", ax=ax)
    ax.set_title("IPS atual por risco no próximo ano")
    ax.set_xlabel("Risco no próximo ano")
    ax.set_ylabel("IPS")
    fig.suptitle("")
    path = save_figure(fig, "05_ips_risco_proximo.png")
    diff = float(grouped.loc[1, "mean"] - grouped.loc[0, "mean"])
    answer = (
        f"Os padroes psicossociais antecedem mudancas de risco: a media de IPS difere em {diff:.2f} "
        f"pontos entre quem entra ou permanece em risco e quem segue fora de risco."
    )
    return path, build_item(
        item_id="5",
        title="Aspectos psicossociais (IPS)",
        answer=answer,
        key_number=f"{diff:.2f}",
        figure=path,
        managerial_takeaway="O bloco psicossocial aparece antes da piora academica e pode ser usado como alerta preventivo.",
        recommended_action="Priorizar suporte psicossocial e contato com familia nos perfis com pior IPS mesmo antes da queda do INDE.",
    )


def build_q6(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    grouped = base.groupby("categoria_ian")["ipp"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax, color="#7d3c98")
    ax.set_title("IPP medio por categoria de IAN")
    ax.set_xlabel("Categoria do IAN")
    ax.set_ylabel("IPP medio")
    path = save_figure(fig, "06_ipp_vs_ian.png")
    gap = float(grouped.max() - grouped.min())
    answer = f"O IPP acompanha a defasagem: a diferenca entre o maior e o menor valor medio de IPP e {gap:.2f}."
    return path, build_item(
        item_id="6",
        title="Aspectos psicopedagogicos (IPP)",
        answer=answer,
        key_number=f"{gap:.2f}",
        figure=path,
        managerial_takeaway="As avaliacoes psicopedagogicas reforcam o diagnostico de defasagem e ajudam a qualificar a priorizacao.",
        recommended_action="Usar IPP para diferenciar alunos com defasagem estrutural de casos mais pontuais de oscilacao academica.",
    )


def build_q7(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    columns = ["ida", "ieg", "iaa", "ips", "ipp", "media_academica", "media_comportamental"]
    correlations = base[columns + ["ipv"]].corr(numeric_only=True)["ipv"].drop("ipv").sort_values(key=np.abs, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    correlations.plot(kind="bar", ax=ax, color=["#1abc9c" if value >= 0 else "#c0392b" for value in correlations.values])
    ax.set_title("Variaveis mais associadas ao IPV")
    ax.set_xlabel("Variavel")
    ax.set_ylabel("Correlacao com IPV")
    path = save_figure(fig, "07_ipv_drivers.png")
    top_name = str(correlations.index[0])
    top_value = float(correlations.iloc[0])
    answer = f"O IPV e mais influenciado por {top_name}, com correlacao de {top_value:.2f}."
    return path, build_item(
        item_id="7",
        title="Ponto de virada (IPV)",
        answer=answer,
        key_number=f"{top_value:.2f}",
        figure=path,
        managerial_takeaway="A virada positiva depende de um conjunto de sinais academicos e comportamentais, nao de um unico evento.",
        recommended_action="Tratar o IPV como indicador de combinacao de sinais e nao apenas como resultado final do ano.",
    )


def build_q8(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    columns = ["inde_atual", "ida", "ieg", "ips", "ipp", "iaa", "ipv", "media_academica", "media_comportamental"]
    corr = base[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    ax.set_title("Heatmap dos indicadores multidimensionais")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = save_figure(fig, "08_multidimensionalidade_heatmap.png")
    inde_rel = corr["inde_atual"].drop("inde_atual").sort_values(key=np.abs, ascending=False)
    answer = (
        f"A combinacao mais alinhada ao desempenho global atual e {inde_rel.index[0]}, "
        f"com correlacao de {float(inde_rel.iloc[0]):.2f} com o INDE."
    )
    return path, build_item(
        item_id="8",
        title="Multidimensionalidade dos indicadores",
        answer=answer,
        key_number=f"{float(inde_rel.iloc[0]):.2f}",
        figure=path,
        managerial_takeaway="O INDE responde a um ecossistema de fatores; olhar um indicador isolado reduz a capacidade de agir cedo.",
        recommended_action="Adotar paines por aluno com visao combinada de engajamento, academico, psicossocial e psicopedagogico.",
    )


def build_q9(training: TrainingArtifacts) -> tuple[Path, dict[str, str]]:
    metrics = training.metrics_holdout.copy()
    plot_metrics = metrics[metrics["model"].isin(["baseline_regra_negocio", training.model_name])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_names = ["recall", "precision", "f2", "pr_auc", "roc_auc"]
    x = np.arange(len(metric_names))
    width = 0.35
    baseline = plot_metrics[plot_metrics["model"] == "baseline_regra_negocio"].iloc[0]
    model_row = plot_metrics[plot_metrics["model"] == training.model_name].iloc[0]
    axes[0].bar(x - width / 2, [baseline[m] for m in metric_names], width=width, label="Baseline", color="#95a5a6")
    axes[0].bar(x + width / 2, [model_row[m] for m in metric_names], width=width, label=training.model_name, color="#2e86de")
    axes[0].set_xticks(x, metric_names, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Comparação baseline x modelo final")
    axes[0].legend()
    confusion = np.array(
        [
            [model_row["true_negative"], model_row["false_positive"]],
            [model_row["false_negative"], model_row["true_positive"]],
        ]
    )
    im = axes[1].imshow(confusion, cmap="Blues")
    axes[1].set_title("Matriz de confusão no holdout")
    axes[1].set_xticks([0, 1], ["Prev. não risco", "Prev. risco"])
    axes[1].set_yticks([0, 1], ["Real não risco", "Real risco"])
    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, int(confusion[row, col]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    path = save_figure(fig, "09_modelo_holdout.png")
    answer = (
        f"No holdout 2023->2024, o modelo {training.model_name} atingiu recall={model_row['recall']:.2f}, "
        f"precision={model_row['precision']:.2f} e F2={model_row['f2']:.2f}."
    )
    return path, build_item(
        item_id="9",
        title="Previsao de risco com Machine Learning",
        answer=answer,
        key_number=f"F2={model_row['f2']:.2f}",
        figure=path,
        managerial_takeaway="O modelo consegue antecipar parte relevante dos casos de risco usando apenas indicadores disponiveis antes do proximo ciclo.",
        recommended_action="Usar a previsao como fila de triagem, sempre combinada com leitura pedagogica e psicossocial da equipe.",
    )


def build_q10(base: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    risk_cycle = base.pivot_table(index="ano_referencia", columns="ciclo_programa", values="risco_atual", aggfunc="mean")
    inde_cycle = base.pivot_table(index="ano_referencia", columns="ciclo_programa", values="inde_atual", aggfunc="mean")
    risk_cycle_plot = apply_month_year_index(risk_cycle)
    inde_cycle_plot = apply_month_year_index(inde_cycle)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    risk_cycle_plot.plot(marker="o", linewidth=2, ax=axes[0])
    axes[0].set_title("Taxa de risco por ciclo do programa")
    axes[0].set_xlabel("Mês/ano de referência")
    axes[0].set_ylabel("Taxa de risco")
    inde_cycle_plot.plot(marker="o", linewidth=2, ax=axes[1])
    axes[1].set_title("INDE medio por ciclo do programa")
    axes[1].set_xlabel("Mês/ano de referência")
    axes[1].set_ylabel("INDE medio")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
    path = save_figure(fig, "10_efetividade_programa.png")
    top_cycle = inde_cycle.loc[2024].sort_values(ascending=False).index[0]
    answer = f"Em 2024, o ciclo com melhor desempenho medio foi {top_cycle}, reforcando a efetividade heterogenea do programa."
    return path, build_item(
        item_id="10",
        title="Efetividade do programa",
        answer=answer,
        key_number=str(top_cycle),
        figure=path,
        managerial_takeaway="O impacto do programa existe, mas nao acontece no mesmo ritmo em todos os ciclos.",
        recommended_action="Customizar metas e cadencia de acompanhamento por ciclo em vez de aplicar a mesma estrategia para toda a base.",
    )


def build_q11(base_pares: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    data = base_pares.copy()
    bins = [-1, 1, 3, 10]
    labels = ["0-1 ano", "2-3 anos", "4+ anos"]
    data["faixa_programa"] = pd.cut(data["anos_no_programa"], bins=bins, labels=labels)
    grouped = data.groupby("faixa_programa", observed=False)[["risco_proximo_ano"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped["risco_proximo_ano"].plot(kind="bar", ax=ax, color="#d35400")
    ax.set_title("Risco no proximo ano por tempo de programa")
    ax.set_xlabel("Tempo de programa")
    ax.set_ylabel("Taxa de risco no proximo ano")
    path = save_figure(fig, "11_insight_extra_tempo_programa.png")
    diff = float(grouped.max().iloc[0] - grouped.min().iloc[0])
    answer = (
        f"O risco de proximo ano varia {format_pct(diff)} entre as faixas de tempo de programa, "
        f"indicando necessidade de acompanhamento segmentado."
    )
    return path, build_item(
        item_id="11",
        title="Tempo de programa e risco futuro",
        answer=answer,
        key_number=format_pct(diff),
        figure=path,
        managerial_takeaway="Tempo de casa altera o padrao de risco, portanto a mesma acao nao serve para veteranos e ingressantes.",
        recommended_action="Segmentar acompanhamento por maturidade no programa, com trilhas especificas para ingressantes e alunos antigos.",
    )


def build_extra_risk_transition(base_pares: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    data = base_pares.copy()
    transition_labels = {
        (0, 0): "Estável sem risco",
        (0, 1): "Entrou em risco",
        (1, 0): "Recuperou",
        (1, 1): "Permaneceu em risco",
    }
    data["transicao_risco"] = list(zip(data["risco_atual"].fillna(0).astype(int), data["risco_proximo_ano"].astype(int)))
    data["transicao_risco"] = data["transicao_risco"].map(transition_labels)
    order = ["Estável sem risco", "Entrou em risco", "Recuperou", "Permaneceu em risco"]
    shares = data["transicao_risco"].value_counts(normalize=True).reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    shares.plot(kind="bar", ax=ax, color=["#2e86de", "#c0392b", "#27ae60", "#e67e22"])
    ax.set_title("Transições de risco entre ciclos")
    ax.set_xlabel("Transição")
    ax.set_ylabel("Participação na base de pares")
    path = save_figure(fig, "12_transicoes_risco.png")
    answer = (
        f"{format_pct(float(shares['Entrou em risco']))} entram em risco no ciclo seguinte, "
        f"enquanto {format_pct(float(shares['Recuperou']))} conseguem se recuperar."
    )
    return path, build_item(
        item_id="E1",
        title="Transicoes de risco ano a ano",
        answer=answer,
        key_number=format_pct(float(shares["Entrou em risco"])),
        figure=path,
        managerial_takeaway="A base exige duas estrategias em paralelo: prevenir novas entradas em risco e recuperar quem ja esta vulneravel.",
        recommended_action="Criar uma fila operacional separando alunos que entram em risco dos que permanecem em risco para intervencoes diferentes.",
    )


def build_extra_delta_inde_alarm(base_pares: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    data = base_pares[base_pares["delta_inde"].notna()].copy()
    bins = [-10, -0.75, -0.25, 0.25, 10]
    labels = ["Queda forte", "Queda leve", "Estavel", "Alta recente"]
    data["faixa_delta_inde"] = pd.cut(data["delta_inde"], bins=bins, labels=labels)
    grouped = data.groupby("faixa_delta_inde", observed=False)["risco_proximo_ano"].mean().reindex(labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped.plot(kind="bar", ax=ax, color=["#c0392b", "#d98880", "#7f8c8d", "#27ae60"])
    ax.set_title("Risco no proximo ano por tendencia recente do INDE")
    ax.set_xlabel("Faixa de delta INDE")
    ax.set_ylabel("Taxa de risco no proximo ano")
    path = save_figure(fig, "13_delta_inde_alerta.png")
    gap = float(grouped.max() - grouped.min())
    answer = (
        f"A diferenca entre a pior e a melhor faixa de variacao do INDE e de {format_pct(gap)} "
        f"na taxa de risco do proximo ano."
    )
    return path, build_item(
        item_id="E2",
        title="Queda recente do INDE como alerta preventivo",
        answer=answer,
        key_number=format_pct(gap),
        figure=path,
        managerial_takeaway="A piora recente do INDE funciona como sinal precoce e de facil operacionalizacao para a equipe.",
        recommended_action="Acionar revisao de caso sempre que houver queda relevante de INDE, mesmo antes da consolidacao do risco.",
    )


def build_extra_evaluation_coverage(base_pares: pd.DataFrame) -> tuple[Path, dict[str, str]]:
    data = base_pares[base_pares["n_av"].notna()].copy()
    bins = [-1, 2, 4, 100]
    labels = ["0-2 avaliacoes", "3-4 avaliacoes", "5+ avaliacoes"]
    data["faixa_avaliacoes"] = pd.cut(data["n_av"], bins=bins, labels=labels)
    grouped = data.groupby("faixa_avaliacoes", observed=False)["risco_proximo_ano"].mean().reindex(labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped.plot(kind="bar", ax=ax, color=["#e67e22", "#5dade2", "#1f618d"])
    ax.set_title("Risco no proximo ano por cobertura de avaliacoes")
    ax.set_xlabel("Quantidade de avaliacoes registradas")
    ax.set_ylabel("Taxa de risco no proximo ano")
    path = save_figure(fig, "14_cobertura_avaliacoes.png")
    gap = float(grouped.max() - grouped.min())
    answer = (
        f"A taxa de risco varia {format_pct(gap)} entre alunos com baixa e alta cobertura de avaliacoes, "
        f"sugerindo que monitoramento insuficiente caminha junto com maior vulnerabilidade."
    )
    return path, build_item(
        item_id="E3",
        title="Cobertura de avaliacoes como sinal operacional",
        answer=answer,
        key_number=format_pct(gap),
        figure=path,
        managerial_takeaway="Menor cobertura avaliativa deixa a equipe mais cega e coincide com maior risco futuro.",
        recommended_action="Garantir nivel minimo de avaliacoes por aluno para reduzir pontos cegos na gestao pedagogica.",
    )


def build_analytics(base: pd.DataFrame, base_pares: pd.DataFrame, training: TrainingArtifacts) -> AnalyticsArtifacts:
    q_builders = [
        build_q1,
        build_q2,
        build_q3,
        build_q4,
        build_q5,
        build_q6,
        build_q7,
        build_q8,
        build_q9,
        build_q10,
        build_q11,
    ]
    extra_builders = [
        build_extra_risk_transition,
        build_extra_delta_inde_alarm,
        build_extra_evaluation_coverage,
    ]

    figure_paths: dict[str, Path] = {}
    qna_items: list[dict[str, str]] = []
    extra_items: list[dict[str, str]] = []

    for index, builder in enumerate(q_builders, start=1):
        if builder in {build_q5, build_q11}:
            path, item = builder(base_pares)
        elif builder is build_q9:
            path, item = builder(training)
        else:
            path, item = builder(base)
        figure_paths[f"q{index}"] = path
        qna_items.append(item)

    for index, builder in enumerate(extra_builders, start=1):
        path, item = builder(base_pares)
        figure_paths[f"extra_{index}"] = path
        extra_items.append(item)

    latest_year = int(base["ano_referencia"].max())
    latest_base = base.loc[base["ano_referencia"] == latest_year]
    selected_model = training.metrics_holdout[training.metrics_holdout["model"] == training.model_name].iloc[0]
    resumo_painel = {
        "base_registros": str(len(base)),
        "alunos_unicos": str(base["ra"].nunique()),
        "pares_modelagem": str(len(base_pares)),
        "ano_mais_recente": str(latest_year),
        "ian_medio_mais_recente": f"{latest_base['ian'].mean():.2f}",
        "taxa_risco_mais_recente": format_pct(float(latest_base["risco_atual"].mean())),
        "variacao_ida_2022_2024": format_delta(float(base.loc[base["ano_referencia"] == 2024, "ida"].mean() - base.loc[base["ano_referencia"] == 2022, "ida"].mean())),
        "modelo_final": training.model_name,
        "threshold_final": f"{training.threshold:.2f}",
        "recall_holdout": f"{float(selected_model['recall']):.2f}",
        "precision_holdout": f"{float(selected_model['precision']):.2f}",
        "f2_holdout": f"{float(selected_model['f2']):.2f}",
    }
    return AnalyticsArtifacts(
        figure_paths=figure_paths,
        qna_items=qna_items,
        extra_items=extra_items,
        resumo_painel=resumo_painel,
    )
