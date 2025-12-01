import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def compute_counts_and_means(n_arms, chosen_arms, rewards):
    """
    A partir do histórico, calcula:
    - tentativas por gênero
    - likes por gênero
    - média de likes por gênero
    """
    chosen_arms = np.array(chosen_arms, dtype=int)
    rewards = np.array(rewards, dtype=float)

    counts = np.zeros(n_arms, dtype=int)
    likes = np.zeros(n_arms, dtype=float)
    means = np.zeros(n_arms, dtype=float)

    for arm in range(n_arms):
        mask = (chosen_arms == arm)
        counts[arm] = mask.sum()
        if counts[arm] > 0:
            likes[arm] = rewards[mask].sum()
            means[arm] = likes[arm] / counts[arm]
        else:
            likes[arm] = 0
            means[arm] = 0.0

    return counts, likes, means


# def fig_cumulative_reward_all(resultados, titulo="Recompensa acumulada"):
#     fig, ax = plt.subplots(figsize=(7, 4))
#     for nome, res in resultados.items():
#         ax.plot(res["cumulative_reward"], label=nome)
#     ax.set_xlabel("Rodadas")
#     ax.set_ylabel("Likes acumulados")
#     ax.set_title(titulo)
#     ax.grid(True)
#     ax.legend()
#     fig.tight_layout()
#     return fig


# def fig_pct_optimal_all(resultados, titulo="% de escolhas do melhor gênero"):
#     fig, ax = plt.subplots(figsize=(7, 4))
#     for nome, res in resultados.items():
#         ax.plot(res["pct_optimal"], label=nome)
#     ax.set_xlabel("Rodadas")
#     ax.set_ylabel("% de vezes que o melhor gênero foi escolhido")
#     ax.set_title(titulo)
#     ax.grid(True)
#     ax.legend()
#     fig.tight_layout()
#     return fig


# def fig_genre_usage(genres, chosen_arms, titulo="Uso de cada gênero"):
#     counts = np.bincount(chosen_arms, minlength=len(genres))
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.bar(genres, counts)
#     ax.set_xlabel("Gênero")
#     ax.set_ylabel("Nº de recomendações")
#     ax.set_title(titulo)
#     ax.grid(True, axis="y", linestyle="--", alpha=0.5)
#     fig.tight_layout()
#     return fig


# def fig_mean_estimates(genres, chosen_arms, rewards, titulo="Média de likes por gênero"):
#     counts, likes, means = compute_counts_and_means(len(genres), chosen_arms, rewards)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.bar(genres, means)
#     ax.set_xlabel("Gênero")
#     ax.set_ylabel("Média de likes (estimada)")
#     ax.set_title(titulo)
#     ax.grid(True, axis="y", linestyle="--", alpha=0.5)
#     fig.tight_layout()
#     return fig

def fig_cumulative_reward_all(resultados, titulo="Recompensa acumulada"):
    fig = go.Figure()

    for nome, res in resultados.items():
        fig.add_trace(
            go.Scatter(
                y=res["cumulative_reward"],
                mode="lines",
                name=nome
            )
        )

    fig.update_layout(
        title=titulo,
        xaxis_title="Rodadas",
        yaxis_title="Likes acumulados",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",  # transparente para acompanhar o app
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Algoritmos",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    return fig


def fig_pct_optimal_all(resultados, titulo="% de escolhas do melhor gênero"):
    fig = go.Figure()

    for nome, res in resultados.items():
        fig.add_trace(
            go.Scatter(
                y=res["pct_optimal"],
                mode="lines",
                name=nome
            )
        )

    fig.update_layout(
        title=titulo,
        xaxis_title="Rodadas",
        yaxis_title="% de vezes que o melhor gênero foi escolhido",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Algoritmos",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 1])

    return fig


def fig_genre_usage(genres, chosen_arms, titulo="Uso de cada gênero"):
    counts = np.bincount(chosen_arms, minlength=len(genres))
    total = counts.sum()
    proporcoes = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    fig = go.Figure(
        data=[
            go.Bar(
                x=genres,
                y=proporcoes,
                name="Proporção"
            )
        ]
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="Gênero",
        yaxis_title="Proporção de recomendações",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )

    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 1])

    return fig


def fig_mean_estimates(genres, chosen_arms, rewards, titulo="Média de likes por gênero"):
    counts, likes, means = compute_counts_and_means(len(genres), chosen_arms, rewards)

    fig = go.Figure(
        data=[
            go.Bar(
                x=genres,
                y=means,
                name="Média estimada"
            )
        ]
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="Gênero",
        yaxis_title="Média de likes (estimada)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )

    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 1])

    return fig