from numpy import zeros, random, argmax


class EpsilonGreedyRecommender:
    """
    Algoritmo epsilon-greedy.

    epsilon = probabilidade de EXPLORAR (escolher um braço aleatório).
    (1 - epsilon) = probabilidade de EXPLORAR O MELHOR conhecido
    (escolher o braço com maior média de recompensa).
    """
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon  # Probabilidade de exploração
        self.counts = zeros(n_arms)  # Número de vezes que cada filme foi recomendado
        self.values = zeros(n_arms)  # Valor esperado de recompensa para cada filme

    def select_arm(self):
        # Com probabilidade epsilon, fazemos uma escolha aleatória (exploração)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms)
        # Caso contrário, fazemos explotação
        return argmax(self.values)

    def update(self, chosen_arm, reward):
        # Atualiza as estimativas do braço selecionado com base na recompensa observada
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value