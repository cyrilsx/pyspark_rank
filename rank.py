from logging import DEBUG

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.rdd import RDD

from article import parse_line
import logging

logger = logging.getLogger('ranking')
logger.setLevel(DEBUG)

LANGS = ["JavaScript", "Java", "PHP", "Python", "C#", "C++", "Ruby", "CSS",
         "Objective-C", "Perl", "Scala", "Haskell", "MATLAB", "Clojure", "Groovy"]


def load():
    """
    This function should prepare local configuration and initiate a SparkContext.
    Then a file must read and the String RDD must be transformed to a Article RDD
    :return: an RDD of Articles

    :see: Hint: use article.parseLine for the transformation
    """
    logger.info("Prepare spark contest and load data")
    conf = SparkConf().setMaster('local').setAppName("Ranking App")\
        .set("spark.executor.memory", "3g").set("spark.driver.memory", "3g").set("spark.python.worker.memory", "3g") \
        .set("spark.driver.maxResultsSize", 0)
    sc = SparkContext(conf=conf)
    return sc.textFile("data/wikipedia.dat").map(lambda x: parse_line(x))


def occurrence_of_word(word: str, rdd: RDD):
    """
    This function should count the number of occurrence of a word

    >>> rdd = sc.parallelize([Article("art1", "an example just to try"),Article("art2", "another example")])
    >>> occurrence_of_word("example", rdd)
    2
    >>> occurrence_of_word("another", rdd)
    1
    :param word: the expected word
    :param rdd: dataset of articles
    :return: the number of occurrence of the word
    """
    return rdd.filter(lambda a: word in a.content.split(" ")) \
        .aggregate(0, lambda x, y: x + 1, lambda x, y: x + y)


def native_rank(words: list, rdd: RDD):
    """
    This function uses the function occurrence_of_word. This is a straight forward algorithm.
    The result should be sorted from the higher rank to the lowest.
    >>> rdd = sc.parallelize([Article("art1", "an example just to try"),Article("art2", "another example")])
    >>> native_rank(["example", "another", "nothing]", rdd)
    [("example", 2), ("another", 1), ("nothing", 0)]

    :param words: list of word we would like to rank
    :param rdd: dataset of articles
    :return: list of pair (word, nb occ)
    """
    result = []
    for word in words:
        result.append((word, occurrence_of_word(word, rdd)))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def filter_words(words, article):
    result = []
    for word in words:
        if article.mentions_word(word):
            result.append((word, article))
    return result


def make_index(words: list, rdd: RDD):
    """
    This function return a RDD containing for each word, the list of article where it's
    mentioned.
    :param words:
    :param rdd:
    :return: the word with the list of article containing this word
    """
    return rdd.flatMap(lambda article: filter_words(words, article)).groupByKey()


def rank_using_reverted_index(index: RDD):
    """

    :param index:
    :return:
    """
    return index.map(lambda lang_article: (lang_article[0], len(lang_article[1]))) \
        .sortBy(lambda k: k[1], ascending=False) \
        .collect()


def rank_reduce_by_key(words: list, rdd: RDD):
    """
    This implementation combine index and computation using `reduceByKey`.
    :param words: list of word we would like to rank
    :param rdd: dataset of articles
    :return: list of pair (word, nb occ)
    """
    return rdd.flatMap(lambda article: filter_words(words, article)) \
        .map(lambda a: (a[0], 1))\
        .reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda k: k[1], ascending=False) \
        .collect()


def main():
    logger.info("Start ranking exercice")
    rdd = load()
    occ_java = occurrence_of_word("Java", rdd)
    logger.info("Nb occurence %d" % occ_java)
    print("Nb occurence %d" % occ_java)
    val = native_rank(LANGS, rdd)
    print(val)
    val = rank_using_reverted_index(make_index(LANGS, rdd))
    print(val)
    val = rank_reduce_by_key(LANGS, rdd)
    print(val)


if __name__ == '__main__':
    main()
