from plugins.base import BaseAnalyzer


class FirstWordTitleAnalyzer(BaseAnalyzer):
    """Подсчёт отношения количества первых слов предложений, начинающихся с большой буквы и с маленькой."""

    def __init__(self):
        self.title_words = 0
        self.total_words = 0

    def analyze(self, text):
        for sentence in text.iter_sent():
            first_word = sentence[0]
            if first_word.istitle():
                self.title_words += 1
            self.total_words += 1

    def get_info(self):
        ratio = self.title_words / self.total_words
        return {"title_words_ratio": ratio, "non_title_words_ratio": 1 - ratio}
