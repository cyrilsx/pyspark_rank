Spark Counting word with Python
===
This is some exercise of Spark course in coursera rewriting for python developer. 

Objectives
---
The objectives is to rank some words for a provided dataset.
The dataset in under data/wikipedia

In this exercise, we are going to try 3 different implementations.



1-Initiate the RDD
---
In `rank.py`, complete the function `load` in order to instantiate the SparkContext and load the
data from `data/wikipedia`.

In order to transform a line to an article, you need to use the method `parse_line` from `article.py`.

2-Count the number of occurrence
---
In `rank.py`, complete the function `occurrence_of_word(word: str, rdd: RDD)`.
It should count the number of time the word is mentioned in the dataset. 
See below for some example
```
>>> rdd = sc.parallelize([Article("art1", "an example just to try"),Article("art2", "another example")])
>>> occurrence_of_word("example", rdd)
2
>>> occurrence_of_word("another", rdd)
1 
```

3-Implementation of the rank using occurrence_of_word
---
In `rank.py`, complete the function `native_rank(words: list, rdd: RDD)`. This function 
must use the function occurrence_of_word. This must be straight forward.
The result should be sorted from the higher rank to the lowest.
See below an example:
```
>>> rdd = sc.parallelize([Article("art1", "an example just to try"),Article("art2", "another example")])
>>> native_rank(["example", "another", "nothing"], rdd)
[("example", 2), ("another", 1), ("nothing", 0)]
```

4-Implementation using reverted index
---
First, you need to create the index. Implement the function `make_index(words: list, rdd: RDD)`.
This function should return a RDD containing the list of articles for each word.
See example below:
```
>>> rdd = sc.parallelize([Article("art1", "an example just to try"),Article("art2", "another example")])
>>> index(["example", "another"], rdd)
[("example", [Article1, Article2]), ("another", [Article1]]
```
Then using this index, rank words. (complete the function `rank_using_reverted_index(index: RDD)`) 
The result must be the same as 3).

5-Implementation using reduceByKey
---
Let's combine the index and the ranking by using `reduceByKey`. The result must be the same than 3)4).