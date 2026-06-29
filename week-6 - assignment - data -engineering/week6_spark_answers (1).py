from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

spark = SparkSession.builder \
    .appName("week6") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# Q1 - Driver, Cluster Manager, Executor
# Driver is the main process that runs our Spark application. It builds the execution
# plan (DAG), breaks it into stages and tasks, and coordinates with the cluster manager
# to get resources. It also collects results back when an action is called.
# Cluster Manager handles resource allocation across the cluster - decides which nodes
# become executors. Can be YARN, Kubernetes, or Spark Standalone.
# Executors are the actual worker processes running on cluster nodes. They run the tasks
# assigned by the driver, store cached data in memory/disk, and report status back to the driver.


# Q2 - Lazy Evaluation
# Spark doesn't run anything when we write transformations like filter() or select().
# It just records them and builds a logical plan (DAG). Execution only starts when we
# call an action like show() or count(). This lets Spark look at the full plan first
# and optimize it - for example pushing filters earlier so less data gets read overall.


# Q3 - Read CSV
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("data/source.csv")

print("Q3 output:")
df.printSchema()
print("row count:", df.count())


# Q4 - CSV vs Parquet
# CSV is row-based plain text - every value is a string until parsed at read time.
# Parquet is columnar and binary - all values of one column are stored together, and
# it stores the data types and min/max stats per column block.
# This matters because with Parquet, if a query only needs 2 columns out of 20,
# Spark only reads those 2 (column pruning). With CSV it reads everything every time.
# Parquet also lets Spark skip entire data blocks using the stored stats (predicate pushdown).
# File sizes are also much smaller since similar values in a column compress better together.


# Q5 - Select product_id and price for Electronics
q5 = df.select("product_id", "price").filter(df.category == "Electronics")
print("\nQ5 output:")
q5.show()


# Q6 - Rename column and cast price to Double
# Reading without inferSchema so price genuinely comes in as StringType
df_str = spark.read.option("header", True).csv("data/source.csv")

# Using col("price") instead of df_str["price"] to avoid AnalysisException
# when referencing a column across chained transformations
df_revised = df_str.withColumnRenamed("old_name", "new_name") \
                   .withColumn("price", col("price").cast(DoubleType()))

print("\nQ6 output:")
df_revised.printSchema()
df_revised.select("product_id", "new_name", "price").show(5)


# Q7 - Lineage Graph and fault tolerance
# Spark tracks the full chain of transformations that produced each partition - the lineage.
# If a node fails and loses a partition, Spark doesn't restart the whole job.
# It looks at the lineage, figures out which steps produced that partition,
# and recomputes just that partition on another node. The DAG acts as the recovery plan.


# Q8 - Filter Completed orders with amount > 1000
df_orders = df

q8 = df_orders.filter((df_orders.status == "Completed") & (df_orders.amount > 1000))
print("\nQ8 output:")
q8.select("product_id", "status", "amount").show()


# Q9 - Predicate Pushdown in Parquet
# When a query has a filter, Spark pushes it into the file reading layer before data
# enters memory. Parquet stores min/max stats per column per row-group at write time.
# Spark checks those stats first and skips entire row-groups that can't possibly match
# the filter - without decompressing them. So only data that can actually match the
# filter gets loaded into executor memory.


# Q10 - Add final_price column (18% tax)
df_tax = df.withColumn("final_price", df["base_price"] * 1.18)
print("\nQ10 output:")
df_tax.select("product_id", "base_price", "final_price").show(5)


# Q11 - Transformations vs Actions
# Transformations are lazy - they return a new DataFrame and add to the DAG without executing.
# Examples: select(), filter()
# Actions trigger actual execution and return results to the driver or write data out.
# Examples: show(), count()


# setup - creating parquet file from source data so Q12 has something to read from
import os
os.makedirs("path/to", exist_ok=True)
df.select("user_id", "amount", "region").write.mode("overwrite").parquet("path/to/input")


# Q12 - Load parquet, filter out null user_ids, save as CSV
q12_df = spark.read.parquet("path/to/input")
q12_clean = q12_df.filter(q12_df.user_id.isNotNull())

os.makedirs("path/to", exist_ok=True)
q12_clean.write.mode("overwrite").option("header", True).csv("path/to/output")

print("\nQ12 output:")
spark.read.option("header", True).csv("path/to/output").show()


# Q13 - Client Mode vs Cluster Mode
# In client mode the driver runs on the machine that submitted the job (your laptop etc).
# If that machine disconnects the job fails. Good for interactive work and debugging.
# In cluster mode the driver is launched inside the cluster itself, so the job keeps running
# even if you close the terminal. This is what you'd use for production jobs.


# Q14 - Filter region = North OR priority = High
q14 = df.filter((df.region == "North") | (df.priority == "High"))
print("\nQ14 output:")
q14.select("product_id", "region", "priority").show()


# Q15 - show(5) vs collect()
# collect() pulls every row to the driver as a Python list. On a large dataset this
# crashes the driver with an out-of-memory error since the driver can't hold terabytes in RAM.
# show(5) only fetches 5 rows and keeps everything else distributed on the executors.
# It's safe at any dataset size and is the right approach when just exploring data.


spark.stop()
print("\nDone.")
