from random import sample
from pyspark import RDD
from pyspark.sql import DataFrame, SparkSession
import math

from DIForestModel import DIForestModel
from ITree import ITree


class DIForest:
    """
    A class for training Distributed Isolation Forest model.
    :param num_trees: The number of trees to train as in the forest
    :param max_samples: The number of samples to train each tree on.
        The samples will be chosen randomly from the entire samples dataframe
        and assigned to each tree (with repetitions)
    """
    def __init__(self, num_trees: int, max_samples: int):
        self.num_trees = num_trees
        self.max_samples = max_samples
        self._max_depth = math.ceil(math.log2(max_samples))

    def fit(self, spark: SparkSession, samples: DataFrame, threshold: int = 0.5) -> DIForestModel:
        """
        Train a Distributed Isolation Forest.
        The training phase consists of the following steps:
            1. Assign to each tree max_samples samples from the entire samples dataframe
            2. Train each tree on its samples, independently of the other samples
        :param spark: a SparkSession
        :param samples: The samples to train the model on
        :param threshold: The threshold to set to the model. Will be used to determine if an instance is
            an anomaly or not. If The anomaly score of a given sample is larger
            than the threshold, then the sample will be considered an anomaly.
        :return: The trained DIForest model
        """
        # Map each tree id to its assigned samples
        tree_id_to_samples = self._assign_samples(spark, samples)
        max_depth = spark.sparkContext.broadcast(self._max_depth)

        # Build a tree for each of the tree ids and their samples
        trees = tree_id_to_samples.map(
            lambda t:
            ITree.build(t[1], max_depth.value)
        ).collect()

        return DIForestModel(trees, self.max_samples, threshold)

    def _assign_samples(self, spark: SparkSession, samples: DataFrame) -> RDD:
        """
        Assign to each tree the samples that it will train on.
        The assignment steps are:
            1. For each tree, select the indices of the samples that it will train on
            2. Map each sample index to the ids of the trees that will train on that sample
            3. Broadcast the mapping from step #2 as its relatively small size (num_trees * max_samples)
            4. For each selected sample (selected for training on some tree), map it to (tree_id, sample features)
               for each tree it was selected to be trained on
            5. Reduce on tree id
        :param spark: a SparkSession
        :param samples: The samples to assign
        :return: Assignment of samples for each trees (aka a mapping from tree id to the its selected samples)
        """
        tree_ids = spark.sparkContext.parallelize(range(self.num_trees))
        samples_size = samples.count()

        max_samples = spark.sparkContext.broadcast(self.max_samples)

        # Map each tree to the indices of the samples it will be trained on
        tree_to_sampled_rows_indices = tree_ids.map(
            lambda tree_id: (tree_id, sample(range(0, samples_size), max_samples.value))
        )

        # Map each sample index to the ids of the trees that will train on the sample
        row_index_to_tree_ids = tree_to_sampled_rows_indices.flatMap(
          lambda t: [(sample_idx, [t[0]]) for sample_idx in t[1]]
        ).reduceByKey(lambda a, b: a + b).toDF(["row_idx", "tree_ids"])

        # Map to a dictionary
        row_index_to_tree_ids = {row['row_idx']: row['tree_ids'] for row in row_index_to_tree_ids.collect()}

        # Broadcast the sample index to tree id mapping
        row_index_to_tree_ids = spark.sparkContext.broadcast(row_index_to_tree_ids)

        # First filter the samples to only those that were randomly picked by some tree
        # Then map for each tree id all the samples that belong to it
        return samples.select('features').rdd.zipWithIndex().filter(
            lambda row: row[1] in row_index_to_tree_ids.value
        ).flatMap(
              lambda row: [(tree_id, [row[0].features.values]) for tree_id in row_index_to_tree_ids.value[row[1]]]
        ).reduceByKey(lambda a, b: a + b)
