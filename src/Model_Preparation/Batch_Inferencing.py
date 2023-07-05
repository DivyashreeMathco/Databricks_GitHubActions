# Databricks notebook source
# MAGIC %run ../../src/Model_Preparation/Model_Registering

# COMMAND ----------

X_train

# COMMAND ----------

spark_df = spark.createDataFrame(X_train)

# COMMAND ----------

display(spark_df)

# COMMAND ----------

table_path = "dbfs:/mudugantidivyadinesh@gmail.com/delta/wine_data"

# COMMAND ----------

dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

import mlflow.pyfunc
 
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct
 
# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))
 
new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# Each row now has an associated prediction. Note that the xgboost function does not output probabilities by default, so the predictions are not limited to the range [0, 1].
display(new_data)
