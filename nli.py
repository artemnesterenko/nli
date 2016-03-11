import os
import pymorphy2
import nltk
from plugins.base import BaseAnalyzer
from errors import PluginExistsError
from scipy.spatial.distance import cosine
from itertools import chain


class NLIFactory:
    _nltk_dependences = ["punkt"]

    def __init__(self, plugin_dirs=None):
        NLIFactory._download_nltk_dependences()
        default_dir = "plugins"
        dirs = [default_dir]
        if plugin_dirs is not None:
            dirs.extend(plugin_dirs)
        NLIFactory._load_plugins(dirs)
        self.analyzers_cls = BaseAnalyzer.__subclasses__()
        name_dict = dict()
        for cls in self.analyzers_cls:
            if cls.__name__ in name_dict:
                raise PluginExistsError(
                        "Plugin {0} has the same attribute name = \"{1}\" as plugin {2}!".format(cls, cls.__name__,
                                                                                                 name_dict[
                                                                                                     cls.__name__]))
            else:
                name_dict[cls.__name__] = cls
            cls.setup()

    @staticmethod
    def _download_nltk_dependences():
        downloader = nltk.downloader.Downloader()
        for dependence in NLIFactory._nltk_dependences:
            if not downloader.is_installed(dependence):
                downloader.download(dependence)

    @staticmethod
    def _load_plugins(plugin_dirs):
        # Сюда добавляем имена загруженных модулей
        # Перебирем файлы в папке plugins
        for dir_name in plugin_dirs:
            for next_dir, inner_dirs, file_names in os.walk(dir_name):
                for fname in file_names:
                    # Нас интересуют только файлы с расширением .py
                    if fname.endswith(".py"):
                        # Обрежем расширение .py у имени файла
                        module_name = fname[: -3]
                        # Пропустим файлы base.py и __init__.py
                        if module_name != "base" and module_name != "__init__":
                            # Загружаем модуль и добавляем его имя в список загруженных модулей
                            __import__(next_dir.replace("/", ".").replace("\\", ".") + "." + module_name)

    def make_identifier(self):
        return NLIdentifier(self.analyzers_cls)


class NLIdentifier(dict):
    def __init__(self, analyzers_cls):
        super().__init__((cls.__name__, cls()) for cls in analyzers_cls)

    @staticmethod
    def filter_analyzers_names(name_set, white_list, black_list):
        if white_list:
            name_set &= set(white_list)
        if black_list:
            name_set -= set(black_list)
        return name_set

    def fit(self, text, *, white_list=None, black_list=None):
        for analyzer_name in NLIdentifier.filter_analyzers_names(self.keys(), white_list,
                                                                 black_list):
            analyzer = self[analyzer_name]
            analyzer.analyze(text)

    def plain_similarity(self, other, *, white_list=None, black_list=None):
        self_vector = []
        other_vector = []
        analyzers_names = NLIdentifier.filter_analyzers_names(
                set(self.keys()) & set(other), white_list, black_list)
        if not analyzers_names:
            return 0
        for analyzer_name in analyzers_names:
            self_feature = self[analyzer_name].get_info()
            other_feature = other[analyzer_name].get_info()
            features_names = set(self_feature) | set(other_feature)
            for feature_name in features_names:
                self_vector.append(self_feature.get(feature_name, 0))
                other_vector.append(other_feature.get(feature_name, 0))
        return 1 - cosine(self_vector, other_vector)

    def average_similarity(self, other, *, white_list=None, black_list=None):
        analyzers_names = NLIdentifier.filter_analyzers_names(
                set(self) & set(other), white_list, black_list)
        if not analyzers_names:
            return 0
        similarity_list = []
        for analyzer_name in analyzers_names:
            similarity_list.append(self[analyzer_name].get_similarity(other[analyzer_name]))
        return sum(similarity_list) / len(similarity_list)

    def weighed_average_similarity(self, other, *, white_list=None, black_list=None):
        analyzers_names = NLIdentifier.filter_analyzers_names(
                set(self) & set(other), white_list, black_list)
        if not analyzers_names:
            return 0
        similarity_list = []
        for analyzer_name in analyzers_names:
            weight = self[analyzer_name].weight
            similarity = self[analyzer_name].get_similarity(other[analyzer_name])
            similarity_list.append(similarity * weight)
        weight_sum = sum(self[name].weight for name in analyzers_names)
        return sum(similarity_list) / weight_sum


class Text:
    morph = pymorphy2.MorphAnalyzer()
    parsed_words_dict = dict()

    def __init__(self, messages):
        self._data = []
        self.parsed_words = []
        for message in messages:
            msg = []
            for sentence in nltk.sent_tokenize(message):
                words = nltk.word_tokenize(sentence)
                for word in words:
                    self.parsed_words.append(Text.parse_word(word))
                msg.append(words)
            self._data.append(msg)
        self.text = list(chain(*self.iter_msg()))

    def iter_parsed_words(self):
        return iter(self.parsed_words)

    def iter_msg(self):
        return iter(self._data)

    def iter_sent(self):
        return iter(self.text)

    def iter_word(self):
        return chain(*self.iter_sent())

    def iter_symb(self):
        return chain(*self.iter_word())

    @classmethod
    def parse_word(cls, word):
        word = word.lower()
        if word in cls.parsed_words_dict:
            return cls.parsed_words_dict[word]
        parsed_word = cls.morph.parse(word)
        lemma_score = {}
        for form in parsed_word:
            if form.normal_form in lemma_score:
                lemma_score[form.normal_form] += form.score
            else:
                lemma_score[form.normal_form] = form.score
        lemma = max(lemma_score, key=lambda k: lemma_score[k])
        for form in parsed_word:
            if form.normal_form == lemma:
                cls.parsed_words_dict[word] = form
                return form
