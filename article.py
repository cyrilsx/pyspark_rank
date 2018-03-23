class Article:
    """
    A wikipedia article is composed of a title and a content
    """
    def __init__(self, title = "", content=""):
        self.title = title
        self.content = content

    def __str__(self):
        return "%s[%s]" % (self.title, len(self.content))

    def mentions_word(self, word):
        """
        :param word: any valid word
        :return: true if the word in mentioned in the article
        """
        return word in self.content.split(" ")


def parse_line(line:str):
    """

    :param line:
    :return: an article for the line in parameters
    """
    sep = "</title><text>"
    idx = line.index(sep)
    title = line[14:idx]
    text = line[idx + len(sep):-16]
    return Article(title, text)
