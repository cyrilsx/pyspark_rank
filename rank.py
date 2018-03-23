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
    raise NotImplemented


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
    raise NotImplemented


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
    raise NotImplemented


def make_index(words: list, rdd: RDD):
    """
    This function return a RDD containing for each word, the list of article where it's
    mentioned.
    :param words:
    :param rdd:
    :return: the word with the list of article containing this word
    """
    raise NotImplemented


def rank_using_reverted_index(index: RDD):
    """

    :param index:
    :return:
    """
    raise NotImplemented


def rank_reduce_by_key(words: list, rdd: RDD):
    """
    This implementation combine index and computation using `reduceByKey`.
    :param words: list of word we would like to rank
    :param rdd: dataset of articles
    :return: list of pair (word, nb occ)
    """
    raise NotImplemented

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
