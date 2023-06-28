# Databricks notebook source
options = {
  "sfUrl": "gx89677.central-india.azure.snowflakecomputing.com",
  "sfUser": "Divyasnowflake",
  "sfPassword": "Divya@1525",
  "sfDatabase": "MLOPS",
  "sfSchema": "RAW_DATA",
  "sfWarehouse": "COMPUTE_WH"
}

# COMMAND ----------

df1 = spark.read \
    .format("snowflake") \
    .options(**options) \
    .option("query", "select * from white_wine;") \
    .load()

# COMMAND ----------

display(df1)

# COMMAND ----------

df2 = spark.read \
    .format("snowflake") \
    .options(**options) \
    .option("query", "select * from red_wine;") \
    .load()

# COMMAND ----------

display(df2)
