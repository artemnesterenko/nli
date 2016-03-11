from plugins.base import BaseAnalyzer
from nltk.probability import FreqDist
from itertools import combinations


class POSFrequencyAnalyzer:#(BaseAnalyzer):
    """Не используется. Сделало всех слишком похожими."""

    def __init__(self):
        self.pos_freq = FreqDist()

    def get_info(self):
        return self.pos_freq

    def analyze(self, text):
        for word in text.iter_word():
            tags = morph.parse(word)[0].tag.grammemes_cyr
            for items_in_comb in range(1, len(tags) + 1):
                for tag_comb in combinations(tags, items_in_comb):
                    tag_comb = str(sorted(tag_comb))
                    self.pos_freq[tag_comb] += 1

