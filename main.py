from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

from DIForest import DIForest


def main():
    """
    Example usage of DIForest
    """
    spark = SparkSession.builder().appName("DIForest").getOrCreate()
    samples = spark.read.csv("https.csv")
    samples.printSchema()

    assembler = VectorAssembler(
        inputCols=["f1", "f2", "f3"],
        outputCol="features")
    samples = assembler.transform(samples)

    dis_model = DIForest(100, 256).fit(spark, samples)
    predictions = dis_model.transform(spark, samples)
    predictions.show()


if __name__ == '__main__':
    main()
