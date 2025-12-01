from numpy import array, random


class MusicEnvironment:
    """
    Ambiente de recomendação por gêneros musicais.

    Cada "braço" é um gênero. Opcionalmente, o ambiente pode ter
    probabilidades verdadeiras de like (para o modo simulado).
    """
    def __init__(self, genres, probs=None):
        """
        genres: lista de strings com os nomes dos gêneros
        probs: lista com as probabilidades verdadeiras de like (ou None)
        """
        self.genres = list(genres)
        self.n_arms = len(genres)

        if probs is None:
            self.probs = None
        else:
            self.probs = array(probs, dtype=float)

    def has_true_probs(self):
        return self.probs is not None

    def pull(self, arm):
        """
        Recomenda o gênero 'arm' e retorna 1 (like) caso o valor aleatorio
        ser menor que a probabilidade de gostar do genero, 0 (dislike), caso
        contrário. Usando as probabilidades verdadeiras.

        Só deve ser usado no modo simulado.
        """
        if self.probs is None:
            raise ValueError(
                "Este ambiente não tem probabilidades verdadeiras definidas. "
                "Use input humano (modo ao vivo) em vez de env.pull()."
            )

        p = self.probs[arm]

        return 1 if random.rand() < p else 0