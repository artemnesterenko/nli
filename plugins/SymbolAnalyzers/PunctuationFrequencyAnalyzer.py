from plugins.base import BaseAnalyzer
from nltk.probability import FreqDist


class SymbolFrequencyAnalyzer(BaseAnalyzer):
    """Подсчёт отношения количества появлений каждого знака пунктуации к общему количеству символов"""

    def __init__(self):
        self.punct_symbol_counts = FreqDist()
        self.punctuation_symbols = "!?.,:;()"
        self.symbol_count = 0

    def analyze(self, text):
        for symbol in text.iter_symb():
            if symbol in self.punctuation_symbols:
                self.punct_symbol_counts[symbol] += 1
            self.symbol_count += 1

    def get_info(self):
        return {symbol: (count / self.symbol_count) for symbol, count in self.punct_symbol_counts.items()}


