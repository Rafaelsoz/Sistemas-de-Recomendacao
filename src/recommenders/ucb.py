from numpy import zeros, random, sqrt, log, argmax, sum


class UCBRecommender:
    """
    Algoritmo UCB1 (Upper Confidence Bound).
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = zeros(n_arms)  # Número de vezes que cada filme foi recomendado
        self.values = zeros(n_arms)  # Valor esperado de recompensa para cada filme

    def select_arm(self):
        # Algoritmo UCB1
        total_counts = sum(self.counts)
        if total_counts == 0:
            return random.randint(0, self.n_arms)  # Se nenhum filme foi selecionado ainda, escolha aleatoriamente

        ucb_values = self.values + sqrt(2 * log(total_counts) / (self.counts + 1e-5))  # Evitar divisão por zero
        return argmax(ucb_values)

    def update(self, chosen_arm, reward):
        # Atualiza as estimativas do braço selecionado com base na recompensa observada
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value