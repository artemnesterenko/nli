from scipy.spatial.distance import cosine


class BaseAnalyzer:
    weight = 1

    @staticmethod
    def setup():
        pass

    def analyze(self, text):
        pass

    def get_info(self):
        return {'base_info': 0}

    def get_similarity(self, other):
        self_info = self.get_info()
        other_info = other.get_info()
        feature_names = set(self_info) | set(other_info)
        self_vector = []
        other_vector = []
        for feature in feature_names:
            self_vector.append(self_info[feature] if feature in self_info else 0)
            other_vector.append(other_info[feature] if feature in other_info else 0)
        return 1 - cosine(self_vector, other_vector)

    def __str__(self):
        return str(self.get_info())

    def __repr__(self):
        return str(self)
