from numpy import random


class RandomRecommender:
    """
    Algortimo que escolhe gêneros totalmente aleatoriamente.
    Serve como baseline / comparação.
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms

    def select_arm(self):
        return random.randint(self.n_arms)

    def update(self, arm, reward):
        # Não aprende nada
        pass