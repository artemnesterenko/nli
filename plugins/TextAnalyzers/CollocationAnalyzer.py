from plugins.base import BaseAnalyzer
from nltk.metrics.association import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder


class CollocationAnalyzer(BaseAnalyzer):
    weight = 2

    def __init__(self):
        self.bigram_finder = None
        self.trigram_finder = None
        self.bigram_measures = BigramAssocMeasures()
        self.trigram_measures = TrigramAssocMeasures()
        self.bigram_min_freq = 10
        self.trigram_min_freq = 10
        self.collocations = set()
        self.bigram_collocation_number = 100
        self.trigram_collocation_number = 100
        self.min_word_len = 3

    def analyze(self, text):
        normalized_words = (word.normal_form for word in text.parsed_words)
        self.trigram_finder = TrigramCollocationFinder.from_words(normalized_words)
        self.bigram_finder = self.trigram_finder.bigram_finder()
        self.trigram_finder.apply_freq_filter(self.trigram_min_freq)
        self.bigram_finder.apply_freq_filter(self.bigram_min_freq)

        def filter_func(w):
            return len(w) < self.min_word_len

        self.trigram_finder.apply_word_filter(filter_func)
        self.bigram_finder.apply_word_filter(filter_func)
        bigram_collocations = self.bigram_finder.nbest(
                self.bigram_measures.likelihood_ratio, self.bigram_collocation_number)
        self.collocations.update(bigram_collocations)
        trigram_collocations = self.trigram_finder.nbest(
                self.trigram_measures.likelihood_ratio, self.trigram_collocation_number)
        self.collocations.update(trigram_collocations)

    def get_info(self):
        return {collocation: 1 for collocation in self.collocations}

    def get_similarity(self, other):
        union = self.collocations | other.collocations
        if not union:
            return 0
        intersection = self.collocations & other.collocations
        return len(intersection) / len(union)

