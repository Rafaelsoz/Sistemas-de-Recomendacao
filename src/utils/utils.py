import numpy as np

def simulate(env, algorithm, n_rounds=200):
    """
    MODO 1: SIMULADO

    Roda n_rounds de interação entre o ambiente (com probs verdadeiras)
    e o algoritmo.

    Retorna um dicionário com:
    - rewards: recompensas (0 ou 1) em cada rodada
    - chosen_arms: índice do gênero escolhido em cada rodada
    - cumulative_reward: likes acumulados ao longo das rodadas
    - pct_optimal: % de vezes em que o melhor gênero foi escolhido
                   (só é calculado se env.probs existe)
    """
    n_arms = env.n_arms
    rewards = np.zeros(n_rounds)
    chosen_arms = np.zeros(n_rounds, dtype=int)
    cumulative_reward = np.zeros(n_rounds)
    pct_optimal = np.zeros(n_rounds)

    if env.has_true_probs():
        best_arm = int(np.argmax(env.probs))
        optimal_choices = np.zeros(n_rounds)
    else:
        raise('Value Error')

    for t in range(n_rounds):
        arm = algorithm.select_arm()
        reward = env.pull(arm)  # usa as probabilidades verdadeiras
        algorithm.update(arm, reward)

        rewards[t] = reward
        chosen_arms[t] = arm
        cumulative_reward[t] = rewards[:t+1].sum()

        if best_arm is not None:
            if arm == best_arm:
                optimal_choices[t] = 1
            pct_optimal[t] = optimal_choices[:t+1].sum() / (t + 1)
        else:
            pct_optimal[t] = np.nan

    return {
        "rewards": rewards,
        "chosen_arms": chosen_arms,
        "cumulative_reward": cumulative_reward,
        "pct_optimal": pct_optimal,
    }
