from plugins.base import BaseAnalyzer
from nltk.probability import FreqDist


class SentenceSymbolProbabilityDistributionAnalyzer(BaseAnalyzer):
    """Создаёт информационный источник, выдающий буквы. Сравнение происходит по тем парам букв вида (a)->(b),
    для которых выроятность появления буквы b после a больше средней вероятности появления других букв после a."""

    def __init__(self):
        self.begin_symbol = "begin"
        self.end_symbol = "end"
        self.source_of_information = {self.begin_symbol: FreqDist()}

    def analyze(self, text):
        for sentence in text.iter_sent():
            marked_sentence = sentence + [self.end_symbol]
            prev_symbol = self.begin_symbol
            for word in marked_sentence:
                for symbol in word:
                    if prev_symbol not in self.source_of_information:
                        self.source_of_information[prev_symbol] = FreqDist()
                    prev_symbol_dict = self.source_of_information[prev_symbol]
                    prev_symbol_dict[symbol] += 1
                    prev_symbol = symbol

    def get_info(self):
        info = {}
        for from_symbol, to_symbols in self.source_of_information.items():
            to_symbols_sum = sum(to_symbols.values())
            average = to_symbols_sum / len(to_symbols)
            for to_symbol, count in to_symbols.items():
                if count > average:
                    info["(%s)->(%s)" % (from_symbol, to_symbol)] = count / to_symbols_sum
        return info


