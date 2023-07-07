# Databricks notebook source
# MAGIC %run ../../src/Data_preparation/snowflake_connectivity.py

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df1 = df1.toPandas()

# COMMAND ----------

df1 = df1.rename(columns=lambda x: x.replace('"', ''))

# COMMAND ----------

df1

# COMMAND ----------

df2 = df2.toPandas()

# COMMAND ----------

df2 = df2.rename(columns=lambda x: x.replace('"', ''))

# COMMAND ----------

df2

# COMMAND ----------

df2['is_red'] = 1
df1['is_red'] = 0

# COMMAND ----------

df2

# COMMAND ----------

df1

# COMMAND ----------

#df1.columns = df1.columns.str.lower()

# COMMAND ----------

#df2.columns = df2.columns.str.lower()

# COMMAND ----------

data = pd.concat([df2, df1], axis=0)

# COMMAND ----------

data

# COMMAND ----------

data['quality'].unique()

# COMMAND ----------

data['quality']

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

data

# COMMAND ----------

data['quality']

# COMMAND ----------

import matplotlib.pyplot as plt
 
dims = (3, 4)
 
f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

data.isna().any()

# COMMAND ----------

type(data)

# COMMAND ----------

new_data = spark.createDataFrame(data)

# COMMAND ----------

new_data.write.format("snowflake").option("dbtable", "EDA_DATA").option("sfUrl", "gx89677.central-india.azure.snowflakecomputing.com").option("sfUser", "Divyasnowflake").option("sfPassword", "Divya@1525").option("sfDatabase", "MLOPS").option("sfSchema", "REFINED_DATA").option("sfWarehouse", "COMPUTE_WH").mode("append").save()
