from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import math
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array


class DIForestModel:
    """
    The trained Distributed Isolation Forest model.
    Can be used to predict which samples are anomalies and which are normal instances
    :param trees: The trained trees
    :param num_samples: The number of samples that the trees were trained on
    :param threshold: A threshold for the anomaly score that will be used to determine
        if a sample is an anomaly or not. If The anomaly score of a given sample is larger
        than the threshold, then the sample will be considered an anomaly.
    """
    def __init__(self, trees: List[dict], num_samples: int, threshold: int = 0.5):
        self._trees = trees
        self._threshold = threshold
        self._num_samples = num_samples

    def transform(self, spark: SparkSession, samples: DataFrame) -> DataFrame:
        """
        For each sample in the samples dataframe, predict if its an outlier or a normal instance
        :param spark: a SparkSession
        :param samples: The dataframe of the samples to predict
        :return: The sampkes dataframe with an additional 2 columns - anomalyScore and prediction,
            where anomalyScore is the calculated anomaly score for the sample,
            and prediction is a boolean column, with True = anomaly, False = normal instance
        """
        # TODO: Validate samples features column
        num_samples = spark.sparkContext.broadcast(self._num_samples)
        trees = spark.sparkContext.broadcast(self._trees)

        predictions = samples.withColumn("featuresArray", vector_to_array(col("features")))
        predictions.printSchema()

        # Add anomaly scores as a column to the DF
        predictions = predictions.withColumn(
            "anomalyScore",
            udf(
                lambda sample: DIForestModel._anomaly_score(sample, trees.value,num_samples.value), FloatType()
            )(col("featuresArray")))

        # Add predictions as a column to the DF (based on given threshold)
        predictions = predictions.withColumn("prediction", col("anomalyScore") > self._threshold)
        return predictions

    @staticmethod
    def _anomaly_score(sample: List[float], trees: List[dict], num_samples: int) -> float:
        """
        Calculate the anomaly score of a sample using given trees.
        The anomaly score is calculated in the following way:
        s(sample, num_samples) = 2 ** -(_avg_path_length(sample) / c(num_samples))
        :param sample: The sample to calculate the anomaly score for
        :param trees: The trees to use for the calculation
        :param num_samples: The number of samples that the trees were trained on
        :return: The anomaly score of the sample
        """
        avg_path_length = DIForestModel._avg_path_length(trees, sample)
        return math.pow(2, -(avg_path_length / DIForestModel._c(num_samples)))

    @staticmethod
    def _avg_path_length(trees: List[dict], sample: List[float]) -> float:
        """
        Calculate the average path length of a sample across all given trees
        :param trees: The trees on which to calculate the path length for the sample
        :param sample: The sample to calculate the average path length for
        :return: The average path length of sample on the given trees
        """
        return sum([DIForestModel._path_length(sample, tree) for tree in trees]) / len(trees)

    @staticmethod
    def _path_length(sample: List[float], curr_node: dict, curr_path_length: int = 0) -> int:
        """
        Calculate the path length of a sample in a given tree recursively. The method evaluates
        the sample according to the curr_node. If the curr_node is an external node, then
        it will return the depth of the current node plus c(n), where n is the num_samples property of
        the node. Otherwise if the node is internal, the method will evaluate the split attribute against
        the sample, and will decide if the next node should be the right or left one.
        :param sample: The sample to calculate the path length for
        :param curr_node: The current node to evaluate sample on. If
        :param curr_path_length: The current path length
        :return: The calculated path length of the given sample
        """
        if curr_node["type"] == "internal":
            # The node is internal - check the split attribute and compare the sample's value
            # to the split value to decide if we need to go to the left or the right node
            if sample[curr_node["split_attribute_idx"]] < curr_node["split_value"]:
                return DIForestModel._path_length(sample, curr_node["left_node"], curr_path_length + 1)

            return DIForestModel._path_length(sample, curr_node["right_node"], curr_path_length + 1)

        elif curr_node["type"] == "external":
            # External node - calculate the path length and return it
            # c(tree.num_samples) is added to compensate for the un-built tree nodes
            # due to the limit on the tree depth
            return curr_path_length + DIForestModel._c(curr_node["num_samples"])

        raise Exception("Unsupported tree node type!")

    @staticmethod
    def _c(n: int) -> float:
        """
        Calculate the estimation of aver- age h(x) for external node terminations.
        c(n) = 2H(n − 1) − (2(n − 1)/n)
        :param n: the size of the data the tree is trained on
        :return: (n)
        """
        if n > 2:
            # Must wrap with float due to some PySpark issue
            return float(2 * DIForestModel._H(n - 1) - 2 * (n - 1) / n)
        elif n == 2:
            return 1.0
        return 0.0

    @staticmethod
    def _H(i: int) -> float:
        """
        Calculate the harmonic number using the estimation H(i) ~= ln(i) + 0.5772156649
        :param i: the i to calculate the harmonic number for
        :return: H(i)
        """
        return math.log(i) + 0.5772156649
