# Databricks notebook source
# MAGIC %md # Distributed Isolation Forest
# MAGIC 
# MAGIC In this project we implemented a distributed versions of the Isolation Forest algorithm.
# MAGIC We compare our implementation to 3 other implementations:
# MAGIC 1. Sklearn-IForest - non distributed version of Isolation Forest
# MAGIC 2. SynapseML - distributed implementation by Microsoft
# MAGIC 
# MAGIC We compare the performance on the following datasets:
# MAGIC 1. Shuttle
# MAGIC 2. Annthyroid
# MAGIC 3. ForestCover
# MAGIC 4. Arrhythmia
# MAGIC 5. Particle
# MAGIC 6. Yearp

# COMMAND ----------

# MAGIC %md ## Imports

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from matplotlib import pyplot as plt
from sklearn import metrics
from DIForest import DIForest
from DIForestModel import DIForestModel
from sklearn.ensemble import IsolationForest as SKLearnIsolationForest
import pyspark.sql.functions as F
import numpy as np
import pandas as pd
import time

from synapse.ml.isolationforest import *
from synapse.ml.explainers import *
from synapse.ml.core.platform import *

if running_on_synapse():
    shell = TerminalInteractiveShell.instance()
    shell.define_macro("foo", """a,b=10,20""")
    
from synapse.ml.core.platform import materializing_display as display

# COMMAND ----------

# MAGIC %md ## Utils

# COMMAND ----------

def evaluate_model(model_name, y_true, y_pred, y_score):
    print("Accuracy Score :")
    print(metrics.accuracy_score(y_true, y_pred))
    print()
    
    print("Classification Report :")
    print(metrics.classification_report(y_true, y_pred))
    print()
    
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_true, y_pred))
    print()
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)
    
    plt.subplots(1, figsize=(10,10))
    plt.title(f'{model_name}, AUC={auc}')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()   
    
    precision_recall_auc = metrics.average_precision_score(y_true, y_score)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    
    plt.subplots(1, figsize=(10,10))
    plt.title(f'{model_name}, Precision-Recall AUC={precision_recall_auc}')
    plt.plot(recall, precision)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()  
    


# COMMAND ----------

# MAGIC %md ## Shuttle Dataset

# COMMAND ----------

data = spark.read.table("hive_metastore.default.shuttle")
print("Schema:")
data.printSchema()
print()

samples_count = data.count()
outliers_count = data.where(data["Y"] == 1).count()

print("Count: ", samples_count)
print("Outliers: ", outliers_count)
print("Outliers percentage: ", outliers_count * 100 / samples_count, "%")

input_cols=["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9"]

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data = assembler.transform(data)
data.show()

# COMMAND ----------

# MAGIC %md ### Our implementation

# COMMAND ----------

start_time = time.time()

dis_model = DIForest(100, 256).fit(spark, data)
predictions = dis_model.transform(spark, data)
predictions_df = predictions.select("attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "Y", "outlierScore", "predictedLabel").toPandas()
end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("DISForest", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SKLearn - non distributed implemenation

# COMMAND ----------

non_distributed_data_df = data.toPandas()
non_distributed_data_df = non_distributed_data_df.drop("features", axis=1)

# COMMAND ----------

start_time = time.time()

# Init the model with default parameters
isolation_forest_model = SKLearnIsolationForest(n_estimators=100)

# Fit the model
isolation_forest_model.fit(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9"]])

# Predict
predictions = isolation_forest_model.predict(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9"]])
non_distributed_data_df['outlierScore'] = isolation_forest_model.score_samples(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9"]])
non_distributed_data_df["predictedLabel"] = np.where(predictions == 1, 0, 1)

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SKLearn", non_distributed_data_df["Y"], non_distributed_data_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SynapseML

# COMMAND ----------

start_time = time.time()

# Init IsolationForest like in the SynapseML documentation
contamination = 0.021
num_estimators = 100
max_samples = 256
max_features = 1.0

isolationForest = (
    IsolationForest()
    .setNumEstimators(num_estimators)
    .setMaxSamples(max_samples)
    .setFeaturesCol("features")
    .setPredictionCol("predictedLabel")
    .setScoreCol("outlierScore")
    .setContamination(contamination)
    .setContaminationError(0.01 * contamination)
    .setRandomSeed(1)
)
model = isolationForest.fit(data)
predictions = model.transform(data)

predictions_df = predictions.select("attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "Y", "outlierScore", "predictedLabel").toPandas()

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SynapseML", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ## Annthyroid

# COMMAND ----------

data = spark.read.table("hive_metastore.default.annthyroid")
print("Schema:")
data.printSchema()
print()

samples_count = data.count()
outliers_count = data.where(data["Y"] == 1).count()

print("Count: ", samples_count)
print("Outliers: ", outliers_count)
print("Outliers percentage: ", outliers_count * 100 / samples_count, "%")

input_cols=["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"]

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data = assembler.transform(data)
data.show()

# COMMAND ----------

# MAGIC %md ### Our Implementation

# COMMAND ----------

start_time = time.time()

dis_model = DIForest(100, 256).fit(spark, data)
predictions = dis_model.transform(spark, data)
predictions_df = predictions.select("attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "Y", "outlierScore", "predictedLabel").toPandas()
end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("DISForest", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SKLearn - non distributed implemenation

# COMMAND ----------

non_distributed_data_df = data.toPandas()
non_distributed_data_df = non_distributed_data_df.drop("features", axis=1)

# COMMAND ----------

start_time = time.time()

# Init the model with default parameters
isolation_forest_model = SKLearnIsolationForest(n_estimators=100)

# Fit the model
isolation_forest_model.fit(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"]])

# Predict
predictions = isolation_forest_model.predict(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"]])
non_distributed_data_df['outlierScore'] = isolation_forest_model.score_samples(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"]])
non_distributed_data_df["predictedLabel"] = np.where(predictions == 1, 0, 1)

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SKLearn", non_distributed_data_df["Y"], non_distributed_data_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SynapseML

# COMMAND ----------

start_time = time.time()

# Init IsolationForest like in the SynapseML documentation
contamination = 0.021
num_estimators = 100
max_samples = 256
max_features = 1.0

isolationForest = (
    IsolationForest()
    .setNumEstimators(num_estimators)
    .setMaxSamples(max_samples)
    .setFeaturesCol("features")
    .setPredictionCol("predictedLabel")
    .setScoreCol("outlierScore")
    .setContamination(contamination)
    .setContaminationError(0.01 * contamination)
    .setRandomSeed(1)
)
model = isolationForest.fit(data)
predictions = model.transform(data)

predictions_df = predictions.select("attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "Y", "outlierScore", "predictedLabel").toPandas()

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SynapseML", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ## ForestCover

# COMMAND ----------

data = spark.read.table("hive_metastore.default.cover")
print("Schema:")
data.printSchema()
print()

samples_count = data.count()
outliers_count = data.where(data["Y"] == 1).count()

print("Count: ", samples_count)
print("Outliers: ", outliers_count)
print("Outliers percentage: ", outliers_count * 100 / samples_count, "%")

input_cols=["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "attr10"]

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data = assembler.transform(data)
data.show()

# COMMAND ----------

# MAGIC %md ### Our Implementation

# COMMAND ----------

start_time = time.time()

dis_model = DIForest(100, 256).fit(spark, data)
predictions = dis_model.transform(spark, data)
predictions_df = predictions.select("attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "attr10", "Y", "outlierScore", "predictedLabel").toPandas()
end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("DISForest", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SKLearn - non distributed implementation

# COMMAND ----------

non_distributed_data_df = data.toPandas()
non_distributed_data_df = non_distributed_data_df.drop("features", axis=1)

# COMMAND ----------

start_time = time.time()

# Init the model with default parameters
isolation_forest_model = SKLearnIsolationForest(n_estimators=100)

# Fit the model
isolation_forest_model.fit(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "attr10"]])

# Predict
predictions = isolation_forest_model.predict(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "attr10"]])
non_distributed_data_df['outlierScore'] = isolation_forest_model.score_samples(non_distributed_data_df[["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "attr10"]])
non_distributed_data_df["predictedLabel"] = np.where(predictions == 1, 0, 1)

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SKLearn", non_distributed_data_df["Y"], non_distributed_data_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SynapseML

# COMMAND ----------

start_time = time.time()

# Init IsolationForest like in the SynapseML documentation
contamination = 0.021
num_estimators = 100
max_samples = 256
max_features = 1.0

isolationForest = (
    IsolationForest()
    .setNumEstimators(num_estimators)
    .setMaxSamples(max_samples)
    .setFeaturesCol("features")
    .setPredictionCol("predictedLabel")
    .setScoreCol("outlierScore")
    .setContamination(contamination)
    .setContaminationError(0.01 * contamination)
    .setRandomSeed(1)
)
model = isolationForest.fit(data)
predictions = model.transform(data)

predictions_df = predictions.select("attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7", "attr8", "attr9", "attr10", "Y", "outlierScore", "predictedLabel").toPandas()

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SynapseML", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ## Arrhythmia

# COMMAND ----------

data = spark.read.table("hive_metastore.default.arrhythmia")
print("Schema:")
data.printSchema()
print()

samples_count = data.count()
outliers_count = data.where(data["Y"] == 1).count()

print("Count: ", samples_count)
print("Outliers: ", outliers_count)
print("Outliers percentage: ", outliers_count * 100 / samples_count, "%")

input_cols=[f"attr{i}" for i in range(274)]

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data = assembler.transform(data)
data.show()

# COMMAND ----------

# MAGIC %md ### Our Implementation

# COMMAND ----------

start_time = time.time()

dis_model = DIForest(100, 256).fit(spark, data)
predictions = dis_model.transform(spark, data)
predictions_columns = input_cols + ["Y", "outlierScore", "predictedLabel"]
predictions_df = predictions.select(*predictions_columns).toPandas()
end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("DISForest", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SKLearn - non distributed implementation

# COMMAND ----------

non_distributed_data_df = data.toPandas()
non_distributed_data_df = non_distributed_data_df.drop("features", axis=1)

# COMMAND ----------

start_time = time.time()

# Init the model with default parameters
isolation_forest_model = SKLearnIsolationForest(n_estimators=100)

# Fit the model
isolation_forest_model.fit(non_distributed_data_df[input_cols])

# Predict
predictions = isolation_forest_model.predict(non_distributed_data_df[input_cols])
non_distributed_data_df['outlierScore'] = isolation_forest_model.score_samples(non_distributed_data_df[input_cols])
non_distributed_data_df["predictedLabel"] = np.where(predictions == 1, 0, 1)

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SKLearn", non_distributed_data_df["Y"], non_distributed_data_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SynapseML

# COMMAND ----------

start_time = time.time()

# Init IsolationForest like in the SynapseML documentation
contamination = 0.021
num_estimators = 100
max_samples = 256
max_features = 1.0

isolationForest = (
    IsolationForest()
    .setNumEstimators(num_estimators)
    .setMaxSamples(max_samples)
    .setFeaturesCol("features")
    .setPredictionCol("predictedLabel")
    .setScoreCol("outlierScore")
    .setContamination(contamination)
    .setContaminationError(0.01 * contamination)
    .setRandomSeed(1)
)
model = isolationForest.fit(data)
predictions = model.transform(data)

predictions_columns = input_cols + ["Y", "outlierScore", "predictedLabel"]
predictions_df = predictions.select(*predictions_columns).toPandas()

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SynapseML", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ## Particle

# COMMAND ----------

data = spark.read.table("hive_metastore.default.particle_preproc_csv")
data=data.withColumnRenamed("ground.truth", "ground_truth")
data=data.withColumnRenamed("point.id", "point_id")
data=data.withColumnRenamed("original.label", "original_label")
data=data.withColumnRenamed("V", "V.0")
data = data.withColumn('Y',F.when(data.ground_truth == "anomaly", 1).otherwise(0))

for i in range(50):
    data = data.withColumnRenamed(f"V.{i}", f"V_{i}")

print("Schema:")
data.printSchema()
print()

samples_count = data.count()
outliers_count = data.where(data["Y"] == 1).count()

print("Count: ", samples_count)
print("Outliers: ", outliers_count)
print("Outliers percentage: ", outliers_count * 100 / samples_count, "%")

input_cols=[f"V_{i}" for i in range(50)]

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data = assembler.transform(data)
data.show()

# COMMAND ----------

# MAGIC %md ### Our Implementation

# COMMAND ----------

start_time = time.time()

dis_model = DIForest(100, 256).fit(spark, data)
predictions = dis_model.transform(spark, data)
predictions_columns = input_cols + ["Y", "outlierScore", "predictedLabel"]
predictions_df = predictions.select(*predictions_columns).toPandas()
end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("DISForest", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SKLearn - non distributed implementation

# COMMAND ----------

non_distributed_data_df = data.toPandas()
non_distributed_data_df = non_distributed_data_df.drop("features", axis=1)

# COMMAND ----------

start_time = time.time()

# Init the model with default parameters
isolation_forest_model = SKLearnIsolationForest(n_estimators=100)

# Fit the model
isolation_forest_model.fit(non_distributed_data_df[input_cols])

# Predict
predictions = isolation_forest_model.predict(non_distributed_data_df[input_cols])
non_distributed_data_df['outlierScore'] = isolation_forest_model.score_samples(non_distributed_data_df[input_cols])
non_distributed_data_df["predictedLabel"] = np.where(predictions == 1, 0, 1)

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SKLearn", non_distributed_data_df["Y"], non_distributed_data_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SynapseML

# COMMAND ----------

start_time = time.time()

# Init IsolationForest like in the SynapseML documentation
contamination = 0.021
num_estimators = 100
max_samples = 256
max_features = 1.0

isolationForest = (
    IsolationForest()
    .setNumEstimators(num_estimators)
    .setMaxSamples(max_samples)
    .setFeaturesCol("features")
    .setPredictionCol("predictedLabel")
    .setScoreCol("outlierScore")
    .setContamination(contamination)
    .setContaminationError(0.01 * contamination)
    .setRandomSeed(1)
)
model = isolationForest.fit(data)
predictions = model.transform(data)

predictions_columns = input_cols + ["Y", "outlierScore", "predictedLabel"]
predictions_df = predictions.select(*predictions_columns).toPandas()

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SynapseML", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ## YearP

# COMMAND ----------

data = spark.read.table("hive_metastore.default.yearp")
data=data.withColumnRenamed("ground.truth", "ground_truth")
data=data.withColumnRenamed("point.id", "point_id")
data=data.withColumnRenamed("original.label", "original_label")
data=data.withColumnRenamed("V", "V.0")
data = data.withColumn('Y',F.when(data.ground_truth == "anomaly", 1).otherwise(0))

for i in range(90):
    data = data.withColumnRenamed(f"V.{i}", f"V_{i}")

print("Schema:")
data.printSchema()
print()

samples_count = data.count()
outliers_count = data.where(data["Y"] == 1).count()

print("Count: ", samples_count)
print("Outliers: ", outliers_count)
print("Outliers percentage: ", outliers_count * 100 / samples_count, "%")

input_cols=[f"V_{i}" for i in range(90)]

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data = assembler.transform(data)
data.show()

# COMMAND ----------

# MAGIC %md ### Our Implementation

# COMMAND ----------

start_time = time.time()

dis_model = DIForest(100, 256).fit(spark, data)
predictions = dis_model.transform(spark, data)
predictions_columns = input_cols + ["Y", "outlierScore", "predictedLabel"]
predictions_df = predictions.select(*predictions_columns).toPandas()
end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("DISForest", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SKLearn - non distributed algorithm

# COMMAND ----------

non_distributed_data_df = data.toPandas()
non_distributed_data_df = non_distributed_data_df.drop("features", axis=1)

# COMMAND ----------

start_time = time.time()

# Init the model with default parameters
isolation_forest_model = SKLearnIsolationForest(n_estimators=100)

# Fit the model
isolation_forest_model.fit(non_distributed_data_df[input_cols])

# Predict
predictions = isolation_forest_model.predict(non_distributed_data_df[input_cols])
non_distributed_data_df['outlierScore'] = isolation_forest_model.score_samples(non_distributed_data_df[input_cols])
non_distributed_data_df["predictedLabel"] = np.where(predictions == 1, 0, 1)

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SKLearn", non_distributed_data_df["Y"], non_distributed_data_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------

# MAGIC %md ### SynapseML

# COMMAND ----------

start_time = time.time()

# Init IsolationForest like in the SynapseML documentation
contamination = 0.021
num_estimators = 100
max_samples = 256
max_features = 1.0

isolationForest = (
    IsolationForest()
    .setNumEstimators(num_estimators)
    .setMaxSamples(max_samples)
    .setFeaturesCol("features")
    .setPredictionCol("predictedLabel")
    .setScoreCol("outlierScore")
    .setContamination(contamination)
    .setContaminationError(0.01 * contamination)
    .setRandomSeed(1)
)
model = isolationForest.fit(data)
predictions = model.transform(data)

predictions_columns = input_cols + ["Y", "outlierScore", "predictedLabel"]
predictions_df = predictions.select(*predictions_columns).toPandas()

end_time = time.time()

elapsed_time = end_time - start_time
print('Execution time:', elapsed_time, 'seconds')

evaluate_model("SynapseML", predictions_df["Y"], predictions_df["predictedLabel"], predictions_df["outlierScore"])

# COMMAND ----------


