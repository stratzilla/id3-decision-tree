# ID3 Decision Tree

An implementation of the ID3 decision tree algorithm with options for holdover. Decision trees are a type of supervised learning in the ML/AI space whereby a data set is recursively split on decisions to generate a complex which can classify novel data. Provided the same train and test data each run, it is a deterministic system, although this implementation, using holdout, allows for stochasticity.

In very simple terms, it can be viewed as a questionnaire: when data is fed into the decision tree, a series of "questions" are asked about the data, with every "answer" traversing the tree structure until it reaches a terminal node, a decision. More training data can generate a tree with more or less granularity, with the goal being to generalize the data. Then new data can be classified in a similar way once the tree structure is found.

There are other types of decision trees like CART and C4.5; ID3 is relatively simple in comparison but the others can make for great reading. They all perform the same task: generate a decision tree, although some others may prune the tree or lead to differing results.

# Dependencies

- pandas
- numpy
- Python (3.6+)
- GNU/Linux

`pandas` has subdependencies which this repository also uses. `python3 -m pip install pandas` should install everything this repository needs.

# Execution

You can clone these files to your computer with the below:

` $ git clone https://github.com/stratzilla/id3-decision-tree`

Execute the script as one of the below:

```shell
 $ ./id3_tree.py <File> <Holdout> <Print>
 $ ./id3_tree.py <Train> <Test> <Print>
```

The arguments differ depending on which method of execution you use.

In the first, a single data set is used for both training and testing, divided by some ratio:

```
 <File> -- the .CSV file of examples
 <Holdout> -- the proportion of training examples (0.00..1.00)
 <Print> -- whether to print tree (1 = T, 0 = F)
```

For the second, separate data sets are used instead:

```
 <Train> -- the .CSV file of training examples
 <Test> -- the .CSV file of testing examples
 <Print> -- whether to print tree (1 = T, 0 = F)
```

Using no arguments will remind the user of this. Irrespective of choice, a decision tree will be made and the accuracy of the tree in classification will be outputted to the console. If printing the tree is selected, both a dict format and "pretty" format tree will be shown.

# Data

.CSV data should be formatted like below:

```
F1 F2 F3 F4 F5 F6 ... FN D
 0  0  1  1  0  1 ... 1 1
 1  0  1  1  0  1 ... 0 0
 1  0  0  0  1  1 ... 1 0
 1  1  0  1  1  0 ... 1 1
 ... ... ... ... ... ...
 0  0  1  0  0  0 ... 0 1
```

Where `F1..FN` are some features, and `D` is some decision. The decision column must be the final column.

Within `data/` are some example datasets:

| dataset         | description               | examples | type of testing suggested  |
| --------------- | ------------------------- | -------- | -------------------------- |
| heart-train.csv | SPECT Heart data set      | 80       | Separated (train set)      |
| heart-test.csv  | SPECT Heart data set      | 187      | Separated (test set)       |
| tennis.csv      | "hello world" of ID3      | 14       | Any                        |
| iris.csv        | Classification of flowers | 150      | Any                        |
