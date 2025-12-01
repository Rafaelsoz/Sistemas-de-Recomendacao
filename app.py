import streamlit as st
import numpy as np
import random

from src.utils import *
from src.music_env import *
from src.recommenders import *
from src.plots import *


def compute_counts_and_means(env, chosen_arms, rewards):
    """
    A partir do histórico (chosen_arms, rewards),
    calcula, para cada gênero:

    - quantas vezes foi escolhido
    - quantos likes recebeu
    - média de likes (estimativa da probabilidade)
    """
    n_arms = env.n_arms
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


# ====================================================
#   CONFIGURAÇÃO GERAL DO APP
# ====================================================
st.set_page_config(
    page_title="Bandits Musicais - Epsilon-Greedy e UCB",
    layout="wide"
)

# ====================================================
#   PÁGINA 1 - MODO SIMULADO
# ====================================================

def pagina_modo_simulado():
    st.title("🎧 Modo 1 – Simulado (Gêneros Musicais)")
    st.write(
        """
        Neste modo, o comportamento da turma é **simulado**: 
        cada gênero musical tem uma probabilidade 'verdadeira' de like.
        
        Os algoritmos (Aleatório, Epsilon-Greedy, UCB1) tentam aprender
        qual gênero funciona melhor ao longo das rodadas.
        """
    )

    default_genres = [
        "Pop",
        "Rock",
        "Funk",
        "Sertanejo",
        "Trap",
        "MPB",
        "Forró",
        "Eletrônica",
    ]

    default_probs = {
        "Pop": 0.3,
        "Rock": 0.4,
        "Funk": 0.85,
        "Sertanejo": 0.3,
        "Trap": 0.7,
        "MPB": 0.3,
        "Forró": 0.45,
        "Eletrônica": 0.5,
    }

      # ---------- Probabilidades verdadeiras na PÁGINA
    st.subheader("Probabilidades Verdadeiras de Like")

    cols = st.columns(4)
    true_probs = []
    for i, g in enumerate(default_genres):
        with cols[i % 4]:
            p = st.slider(
                f"{g}",
                min_value=0.0,
                max_value=1.0,
                value=float(default_probs[g]),
                step=0.05,
                key=f"prob_slider_{g}",
            )
            true_probs.append(p)


    # ---------- Configurações na barra lateral ----------
    st.sidebar.header("Configurações – Modo Simulado")

    st.sidebar.markdown("**Gêneros usados:**")
    st.sidebar.write(", ".join(default_genres))

    st.sidebar.markdown("---")
    n_rounds = st.sidebar.slider("Número de rodadas", 50, 2000, 300, 50)
    epsilon = st.sidebar.slider("ε (epsilon) – Epsilon-Greedy", 0.0, 1.0, 0.1, 0.05)

    st.sidebar.markdown("---")
    seed = st.sidebar.number_input("Semente aleatória (para reprodutibilidade)", 0, 10_000, 42)

    # ---------- Botão de simulação ----------
    if st.button("▶ Rodar simulação"):
        # Rodar os três algoritmos com a mesma semente/base
        resultados = {}

        # Aleatório
        np.random.seed(seed)
        rand_env = MusicEnvironment(default_genres, true_probs)
        rand_policy = RandomRecommender(rand_env.n_arms)
        resultados["Aleatório"] = simulate(rand_env, rand_policy, n_rounds=n_rounds)

        # Epsilon-Greedy
        np.random.seed(seed)
        eps_env = MusicEnvironment(default_genres, true_probs)
        eps_policy = EpsilonGreedyRecommender(eps_env.n_arms, epsilon=epsilon)
        resultados[f"Epsilon-Greedy"] = simulate(
            eps_env, eps_policy, n_rounds=n_rounds
        )

        # UCB1
        np.random.seed(seed)
        ucb_env = MusicEnvironment(default_genres, true_probs)
        ucb_policy = UCBRecommender(ucb_env.n_arms)
        resultados[f"UCB1"] = simulate(
            ucb_env, ucb_policy, n_rounds=n_rounds
        )

        st.markdown("---")
        st.subheader("Curvas de aprendizado - Comparação de Algoritmos")

        fig1 = fig_cumulative_reward_all(
            resultados, titulo="Recompensa acumulada – modo simulado"
        )
        fig2 = fig_pct_optimal_all(
            resultados, titulo="% de escolhas do melhor gênero – modo simulado"
        )

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            # st.pyplot(fig1)
            st.plotly_chart(fig1, use_container_width=True)
        with col_l2:
            # st.pyplot(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        st.info(
            "Nota: No começo a uma exploração melhor, por isso"
            "a menos likes acumulados bem como menos escolhas do melhor genêro."
            "\nConforme o algortimo aprende, começa a recomendar o genêro de maior probabiliade / nota."
        )

        # Detalhamento Aleatório
        st.markdown("---")
        st.subheader("Detalhando o comportamento do Aleatório")
        ucb_results = resultados[f"Aleatório"]

        fig3 = fig_genre_usage(
            ucb_env.genres,
            ucb_results["chosen_arms"],
            titulo="Uso de cada gênero – Aleatório",
        )
        fig4 = fig_mean_estimates(
            ucb_env.genres,
            ucb_results["chosen_arms"],
            ucb_results["rewards"],
            titulo="Média de likes estimada – Aleatório",
        )

        # Plots de BARRAS lado a lado
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            # st.pyplot(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        with col_b2:
            # st.pyplot(fig4)
            st.plotly_chart(fig4, use_container_width=True)

        # Detalhamento UCB
        st.subheader("Detalhando o comportamento do UCB1")
        ucb_results = resultados[f"UCB1"]

        fig3 = fig_genre_usage(
            ucb_env.genres,
            ucb_results["chosen_arms"],
            titulo="Uso de cada gênero – UCB1",
        )
        fig4 = fig_mean_estimates(
            ucb_env.genres,
            ucb_results["chosen_arms"],
            ucb_results["rewards"],
            titulo="Média de likes estimada – UCB1",
        )

        # Plots de BARRAS lado a lado
        col_b3, col_b4 = st.columns(2)
        with col_b3:
            # st.pyplot(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        with col_b4:
            # st.pyplot(fig4)
            st.plotly_chart(fig4, use_container_width=True)

        # Epsilon Greedy
        st.subheader("Detalhando o comportamento do Epsilon-Greedy")
        ucb_results = resultados[f"Epsilon-Greedy"]

        fig3 = fig_genre_usage(
            ucb_env.genres,
            ucb_results["chosen_arms"],
            titulo="Uso de cada gênero – Epsilon-Greedy",
        )
        fig4 = fig_mean_estimates(
            ucb_env.genres,
            ucb_results["chosen_arms"],
            ucb_results["rewards"],
            titulo="Média de likes estimada – Epsilon-Greedy",
        )

        # Plots de BARRAS lado a lado
        col_b5, col_b6 = st.columns(2)
        with col_b5:
            # st.pyplot(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        with col_b6:
            # st.pyplot(fig4)
            st.plotly_chart(fig4, use_container_width=True)


        st.info(
            "Algortimos não aleatórios tendem a recomendar mais frequentemente os"
            "genêros com maior probabilidade de like."
        )
    else:
        st.warning("Clique em **Rodar simulação** para ver as curvas.")

    st.markdown("---")
    st.markdown("---")


# ====================================================
#   PÁGINA 2 - AO VIVO COM A TURMA
# ====================================================
def inicializar_estado_ao_vivo():
    if "ao_vivo_inicializado" not in st.session_state:
        st.session_state.ao_vivo_inicializado = True
        st.session_state.genres = [
            "Pop",
            "Rock",
            "Funk",
            "Sertanejo",
            "MPB",
            "Forró",
        ]
        st.session_state.policy = RandomRecommender(len(st.session_state.genres))
        st.session_state.policy_label = "Aleatório"
        st.session_state.chosen_arms = []
        st.session_state.rewards = []
        st.session_state.waiting_feedback = False
        st.session_state.current_arm = None
        st.session_state.current_song_title = None 


def resetar_experimento_ao_vivo(alg_choice, epsilon):
    st.session_state.chosen_arms = []
    st.session_state.rewards = []
    st.session_state.waiting_feedback = False
    st.session_state.current_arm = None
    st.session_state.current_song_title = None

    if alg_choice == "Aleatório":
        st.session_state.policy = RandomRecommender(len(st.session_state.genres))
        st.session_state.policy_label = "Aleatório"
    elif alg_choice == "Epsilon-Greedy":
        st.session_state.policy = EpsilonGreedyRecommender(len(st.session_state.genres), epsilon=epsilon)
        st.session_state.policy_label = f"Epsilon-Greedy"
    else:
        st.session_state.policy = UCBRecommender(len(st.session_state.genres))
        st.session_state.policy_label = f"UCB"


def processar_feedback(reward):
    # atualiza com o feedback da recomendação atual
    arm = st.session_state.current_arm
    st.session_state.policy.update(arm, reward)
    st.session_state.chosen_arms.append(arm)
    st.session_state.rewards.append(reward)

    # Prepara a PRÓXIMA recomendação automaticamente
    next_arm = st.session_state.policy.select_arm()
    st.session_state.current_arm = next_arm
    next_genero = st.session_state.genres[next_arm]
    st.session_state.current_song_title = gerar_nome_musica(next_genero)
    st.session_state.waiting_feedback = True

    # st.rerun()

# def gerar_nome_musica(genero: str) -> str:
#     genero = genero.lower()

#     # Palavras base por gênero (bem simples, mas já fica divertido)
#     palavras = {
#         "pop": {
#             "adjs": ["Perfeito", "Impossível", "Secreto", "Inesquecível", "Eterno"],
#             "subs": ["Amor", "Verão", "Destino", "Coração", "Momento"],
#         },
#         "rock": {
#             "adjs": ["Quebrado", "Selvagem", "Elétrico", "Sombrio", "Rebelde"],
#             "subs": ["Silêncio", "Tempestade", "Grito", "Caos", "Estrada"],
#         },
#         "funk": {
#             "adjs": ["Proibido", "Pesado", "Diferente", "Do Bailão", "Da Quebrada"],
#             "subs": ["Beat", "Bailão", "Tamborzão", "Rolê", "Mandela"],
#         },
#         "sertanejo": {
#             "adjs": ["Velho", "Doído", "Apaixonado", "Solteiro", "Sozinho"],
#             "subs": ["Coração", "Buteco", "Interior", "Pé de Serra", "Saudade"],
#         },
#         "mpb": {
#             "adjs": ["Doce", "Calmo", "Profundo", "Suave", "Antigo"],
#             "subs": ["Mar", "Lua", "Café", "Saudade", "Janela"],
#         },
#         "forró": {
#             "adjs": ["Quente", "Apaixonado", "Arretado", "Do Sertão", "Do Nordeste"],
#             "subs": ["Forró", "Xote", "Arrasta-pé", "São João", "Lua de Mel"],
#         },
#         # fallback genérico
#         "default": {
#             "adjs": ["Novo", "Antigo", "Secreto", "Distante", "Perdido"],
#             "subs": ["Caminho", "Sonho", "Horizonte", "Encontro", "Sinal"],
#         },
#     }

#     if "pop" in genero:
#         base = palavras["pop"]
#     elif "rock" in genero:
#         base = palavras["rock"]
#     elif "funk" in genero:
#         base = palavras["funk"]
#     elif "sertanejo" in genero:
#         base = palavras["sertanejo"]
#     elif "mpb" in genero:
#         base = palavras["mpb"]
#     elif "forró" in genero or "forro" in genero:
#         base = palavras["forró"]
#     else:
#         base = palavras["default"]

#     adj = random.choice(base["adjs"])
#     sub = random.choice(base["subs"])

#     # Alguns padrões de título
#     padrao = random.choice(["adj_sub", "so_sub", "noite_de_sub"])

#     if padrao == "adj_sub":
#         return f"{adj} {sub}"
#     elif padrao == "so_sub":
#         return sub
#     else:  # "noite_de_sub"
#         return f"Noite de {sub}"
    
def gerar_nome_musica(genero: str) -> str:
    genero = genero.lower()

    # Bancos maiores de palavras por gênero (não precisa ser realista, só divertido)
    palavras = {
        "pop": {
            "adjs": ["Perfeito", "Impossível", "Secreto", "Inesquecível", "Eterno", "Doce",
                     "Louco", "Brilhante", "Proibido", "Azul", "Dançante", "Veloz", "Sincero"],
            "subs": ["Amor", "Verão", "Destino", "Coração", "Momento", "Beijo", "Noite",
                     "Segredo", "Mensagem", "Memória", "Sonho", "Festa", "Estrela"],
            "verbs": ["Diz", "Sente", "Chama", "Vem", "Foge", "Volta", "Explode", "Brilha"],
            "lugares": ["Praia", "Cidade", "Pista", "Varanda", "Céu", "Elevador", "Quarto"],
        },
        "rock": {
            "adjs": ["Quebrado", "Selvagem", "Elétrico", "Sombrio", "Rebelde", "Áspero",
                     "Cruel", "Vermelho", "Infernal", "Livre", "Gigante", "Maldito"],
            "subs": ["Silêncio", "Tempestade", "Grito", "Caos", "Estrada", "Motor", "Fumaça",
                     "Cicatriz", "Ruína", "Noite", "Pedra", "Ferro", "Sombra"],
            "verbs": ["Rasga", "Queima", "Late", "Cai", "Sobe", "Quebra", "Urra", "Ressoa"],
            "lugares": ["Garagem", "Asfalto", "Deserto", "Beco", "Palco", "Subsolo"],
        },
        "funk": {
            "adjs": ["Proibido", "Pesado", "Diferente", "Do Bailão", "Da Quebrada", "Malvadão",
                     "Nervoso", "Reluzente", "Estourado", "Safado", "Gelado"],
            "subs": ["Beat", "Bailão", "Tamborzão", "Rolê", "Mandela", "Passinho", "Revoada",
                     "Grave", "Chão", "Fluxo", "Favela", "Vibe"],
            "verbs": ["Desce", "Sobe", "Joga", "Bate", "Rebola", "Encosta", "Gira", "Treme"],
            "lugares": ["Quadra", "Beco", "Baile", "Rua", "Morro", "Piscina", "Pista"],
        },
        "sertanejo": {
            "adjs": ["Velho", "Doído", "Apaixonado", "Solteiro", "Sozinho", "Perdido",
                     "Tristonho", "Bêbado", "Calado", "Valente", "Teimoso"],
            "subs": ["Coração", "Buteco", "Interior", "Pé de Serra", "Saudade", "Estrada",
                     "Paixão", "Chapéu", "Chuva", "Aliança", "Mensagem", "Lembrança"],
            "verbs": ["Chora", "Liga", "Some", "Volta", "Apaixona", "Esquece", "Promete"],
            "lugares": ["Buteco", "Rodeio", "Fazenda", "Cidadezinha", "Estrada de Terra"],
        },
        "mpb": {
            "adjs": ["Doce", "Calmo", "Profundo", "Suave", "Antigo", "Sereno",
                     "Lírico", "Morno", "Cintilante", "Tranquilo", "Vago"],
            "subs": ["Mar", "Lua", "Café", "Saudade", "Janela", "Brisa", "Rua",
                     "Outono", "Poesia", "Chuva", "Silêncio", "Sorriso"],
            "verbs": ["Lembra", "Canta", "Sopra", "Flutua", "Encosta", "Abraça"],
            "lugares": ["Varanda", "Calçada", "Praça", "Rio", "Esquina", "Barzinho"],
        },
        "forró": {
            "adjs": ["Quente", "Apaixonado", "Arretado", "Do Sertão", "Do Nordeste", "Faceiro",
                     "Safadinho", "Bonito", "Vaqueiro", "Matuto"],
            "subs": ["Forró", "Xote", "Arrasta-pé", "São João", "Lua de Mel", "Sanfona",
                     "Fogueira", "Chão", "Pisada", "Cangaço", "Baião"],
            "verbs": ["Arreda", "Chega", "Chama", "Dança", "Vira", "Puxa", "Roda"],
            "lugares": ["Arraiá", "Sertão", "Feira", "Riacho", "Vila", "Fogueira"],
        },
        "default": {
            "adjs": ["Novo", "Antigo", "Secreto", "Distante", "Perdido", "Lindo",
                     "Estranho", "Curioso", "Alto", "Baixo"],
            "subs": ["Caminho", "Sonho", "Horizonte", "Encontro", "Sinal", "Vento",
                     "Noite", "Luz", "Tempo"],
            "verbs": ["Corre", "Chama", "Sobe", "Cai", "Vira", "Some"],
            "lugares": ["Lugar Nenhum", "Qualquer Canto", "Outro Lado", "Aqui"],
        },
    }

    # escolhe o banco certo
    if "pop" in genero:
        base = palavras["pop"]
    elif "rock" in genero:
        base = palavras["rock"]
    elif "funk" in genero:
        base = palavras["funk"]
    elif "sertanejo" in genero:
        base = palavras["sertanejo"]
    elif "mpb" in genero:
        base = palavras["mpb"]
    elif "forró" in genero or "forro" in genero:
        base = palavras["forró"]
    else:
        base = palavras["default"]

    adj = random.choice(base["adjs"])
    sub1 = random.choice(base["subs"])
    sub2 = random.choice(base["subs"])
    verb = random.choice(base["verbs"])
    lugar = random.choice(base["lugares"])

    conectores = ["e", "com", "sem", "contra", "por", "pra", "depois de", "antes de"]
    conn = random.choice(conectores)

    # vários padrões de título
    padroes = [
        lambda: f"{adj} {sub1}",
        lambda: f"{sub1} {conn} {sub2}",
        lambda: f"{verb} {sub1}",
        lambda: f"{sub1} de {lugar}",
        lambda: f"{adj} {sub1} de {lugar}",
        lambda: f"{sub1} na {lugar}",
        lambda: f"{verb} na {lugar}",
        lambda: f"{sub1}: {adj} {sub2}",
        lambda: f"{sub1} ({adj})",
        lambda: f"{adj} {sub1} / {sub2}",
        lambda: f"{sub1} do {adj}",
        lambda: f"{verb} {conn} {sub1}",
    ]

    titulo = random.choice(padroes)()

    # chance pequena de colocar numeral romano no fim (até III)
    romanos = [" I", " II", " III"]
    if random.random() < 0.18:  # ~18% de chance
        titulo += random.choice(romanos)

    return titulo


def pagina_modo_ao_vivo():
    inicializar_estado_ao_vivo()

    st.title("🎤 Ao Vivo com a Turma")
    st.write(
        """
        Neste modo, **a própria turma é o ambiente**:
        o algoritmo recomenda um gênero e a turma responde se gostou ou não.
        
        Vamos compreender como o algoritmo vai aprendendo a preferência da sala.
        """
    )

    # Configurações na barra lateral
    st.sidebar.header("Configurações – Modo Ao Vivo")

    alg_choice = st.sidebar.radio(
        "Algoritmo",
        ["Aleatório", "Epsilon-Greedy", "UCB1"],
    )

    epsilon_live = 0.1

    if alg_choice == "Epsilon-Greedy":
        epsilon_live = st.sidebar.slider("ε (epsilon)", 0.0, 1.0, 0.1, 0.05)

    if st.sidebar.button("🔄 Reiniciar experimento"):
        resetar_experimento_ao_vivo(alg_choice, epsilon_live)
        st.sidebar.success("Experimento reiniciado!")

    st.subheader(f"Algoritmo atual- **{st.session_state.policy_label}**")

    # Botão para gerar a próxima recomendação
    # if st.button("▶ Recomendar próximo gênero"):
    #     if st.session_state.waiting_feedback:
    #         st.warning("Responda primeiro se a turma gostou ou não da recomendação anterior.")
    #     else:
    #         arm = st.session_state.policy.select_arm()
    #         st.session_state.current_arm = arm
    #         genero = st.session_state.genres[arm]
    #         st.session_state.current_song_title = gerar_nome_musica(genero) 
    #         st.session_state.waiting_feedback = True

    if st.button("▶ Iniciar Recomendações", disabled=st.session_state.waiting_feedback):
        arm = st.session_state.policy.select_arm()
        st.session_state.current_arm = arm
        genero = st.session_state.genres[arm]
        st.session_state.current_song_title = gerar_nome_musica(genero)
        st.session_state.waiting_feedback = True

    st.markdown("---")
    # Se estamos esperando feedback, mostrar o gênero recomendado
    if st.session_state.waiting_feedback and st.session_state.current_arm is not None:
        genero = st.session_state.genres[st.session_state.current_arm]
        titulo = st.session_state.current_song_title or gerar_nome_musica(genero)
        st.subheader(f"Recomendação atual: **{genero}** - _\"{titulo}\"_ 🎵")

        # col1, col2 = st.columns(2)
        # with col1:
        #     if st.button("👍 A turma gostou", key="like_button"):
        #         processar_feedback(1)
        # with col2:
        #     if st.button("👎 Não gostou", key="dislike_button"):
        #         processar_feedback(0)
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "👍 Like",
                key="like_button",
                on_click=processar_feedback,
                args=(1,)
            )
        with col2:
            st.button(
                "👎 Dislike",
                key="dislike_button",
                on_click=processar_feedback,
                args=(0,)
            )
    
    music_env = MusicEnvironment(["Pop", "Rock", "Funk", "Sertanejo", "MPB","Forró"])
    # Mostrar resumo e gráficos
    if len(st.session_state.chosen_arms) > 0:
        st.subheader("Resumo até agora")

        n_arms = len(st.session_state.genres)
        counts, likes, means = compute_counts_and_means(
            music_env,
            st.session_state.chosen_arms,
            st.session_state.rewards,
        )

        total_rodadas = len(st.session_state.chosen_arms)
        total_likes = int(sum(st.session_state.rewards))

        st.write(f"- **Rodadas:** {total_rodadas}")
        st.write(f"- **Likes totais:** {total_likes}")

        st.write("**Por gênero:**")
        resumo = []
        for i, g in enumerate(st.session_state.genres):
            resumo.append(
                {
                    "Gênero": g,
                    "Tentativas": counts[i],
                    "Likes": int(likes[i]),
                    "Média de likes": round(means[i], 2),
                }
            )
        st.table(resumo)

        st.markdown("### Gráficos")

        col1, col2 = st.columns(2)
        with col1:
            fig_usage = fig_genre_usage(
                st.session_state.genres,
                st.session_state.chosen_arms,
                titulo="Nº de recomendações por gênero",
            )
            # st.pyplot(fig_usage)
            st.plotly_chart(fig_usage, use_container_width=True)

        with col2:
            fig_means = fig_mean_estimates(
                st.session_state.genres,
                st.session_state.chosen_arms,
                st.session_state.rewards,
                titulo="Média de likes estimada",
            )
            # st.pyplot(fig_means)
            st.plotly_chart(fig_means, use_container_width=True)

        st.info(
            "Você pode pausar em qualquer momento e perguntar para a turma:\n"
            "- Por que certos gêneros foram mais tocados?\n"
            "- O algoritmo ainda está explorando ou já foca mais nos favoritos?"
        )
    else:
        st.warning("Ainda não houve interações.")

    st.markdown("---")
    st.markdown("---")

# ====================================================
#   “MENU” DE PÁGINAS
# ====================================================

pagina = st.sidebar.selectbox(
    "Escolha a página",
    ["Home", "Simulador"],
)

if pagina.startswith("Simulador"):
    pagina_modo_simulado()
else:
    pagina_modo_ao_vivo()