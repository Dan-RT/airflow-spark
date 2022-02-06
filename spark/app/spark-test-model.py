import sys
from pyspark.sql import SparkSession, functions
from pyspark.sql.types import StringType
from modules import moduleExample

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, dayofweek, datediff, hour, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from collections import defaultdict

# Create spark session
spark = (SparkSession
    .builder
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("WARN")

####################################
# Parameters
####################################
csv_file = sys.argv[1]

####################################
# Read CSV Data
####################################
print("######################################")
print("READING CSV FILE")
print("######################################")


df = spark.read.option("delimiter", ";").option("inferschema", "true").csv(csv_file, header = True) #.limit(10000)

#
# Dataset is imbalanced
# We undersample it
#
major_class = df.filter(col("install") == 0)
minor_class = df.filter(col("install") == 1)

ratio = int(major_class.count()/minor_class.count())
sampled_major_class = major_class.sample(False, 1/ratio)

df = sampled_major_class.unionAll(minor_class)

#
# Process Dates
#

df = df.withColumn("day", date_format('timestamp', 'E').alias('dow_string'))
df = df.withColumn("is_weekend", dayofweek("timestamp").isin([1,7]).cast("int"))
df = df.withColumn("diff_last_start", datediff(df.timestamp, df.lastStart))
df = df.withColumn("hour", hour('timestamp').alias('hour'))

def PartOfTheDay(hour):
    if (hour > 4) and (hour <= 8):
        return 'EarlyMorning'
    elif (hour > 8) and (hour <= 12 ):
        return 'Morning'
    elif (hour > 12) and (hour <= 14):
        return'Noon'
    elif (hour > 14) and (hour <= 18):
        return'AfterNoon'
    elif (hour > 18) and (hour <= 20) :
        return 'Eve'
    elif (hour > 20) and (hour <= 24):
        return'Night'
    elif (hour <= 4):
        return'LateNight'

udf_part_of_day = udf(PartOfTheDay, StringType())
df = df.withColumn('partOfDay', udf_part_of_day(df.hour))

#
# Process numerical data
#

#
# Cast the desired columns to double
#
col_to_cast = ["startCount", "viewCount", "clickCount", "installCount", "startCount1d", "startCount7d", "diff_last_start"]
casted_columns = []
for col in col_to_cast:
    df = df.withColumn(col + "_double", df[col].cast("double"))
    casted_columns.append(col + "_double")

imputed_col = [col + "_imputed" for col in casted_columns]

#
# Completes missing values in the dataset, handle NaN
#
imputer_stage = Imputer(inputCols = casted_columns, outputCols = imputed_col)

#
# Prepare the data for StandardScaler
#
assembler_stage = VectorAssembler(inputCols = imputed_col, outputCol = "features_vector_assembled")

#
# Normalize each feature to have unit standard deviation 
#
scaler_stage = StandardScaler(inputCol = "features_vector_assembled", outputCol = "features_scaled", withStd = True, withMean = False)

#
# Run the stages
#
pipeline = Pipeline(stages= [imputer_stage, assembler_stage, scaler_stage])
df = pipeline.fit(df).transform(df)

#
# Drop columns that are no longer useful or categorical column with too much different values 
#
columns = ["id", "timestamp", "lastStart", "campaignId", "sourceGameId", "softwareVersion", "deviceType", "hour"]
df = df.drop(*columns)

def DetermineTypes(df):
    type_columns = defaultdict(list)
    for entry in df.schema.fields:
        type_columns[str(entry.dataType)].append(entry.name)
    return type_columns

type_columns = DetermineTypes(df)
string_columns = [var for var in type_columns["StringType"]]


#
# Process NaN values by replacing them by "missing"
#
missing_string_data = {}
for var in string_columns:
    missing_string_data[var] = "missing"
df = df.fillna(missing_string_data)

#
# Prepare data for OneHotEncoder
#
string_indexer_stage = [StringIndexer(inputCol = col, outputCol = col + "_string_encoded") for col in string_columns]

#
# Apply OneHotEncoding to categorical values
#
one_hot_encoder_stage = [OneHotEncoder(inputCol = col + "_string_encoded", outputCol = col + "_one_hot") for col in string_columns]

#
# Run the stages
#
pipeline = Pipeline(stages = string_indexer_stage + one_hot_encoder_stage)
df = pipeline.fit(df).transform(df)

print("######################################")
print("PRINTING 10 ROWS OF SAMPLE DF")
print("######################################")

df.show(10)

#
# Merge all the preprocessed columns into column "features"
#
features = ["features_scaled"] + [col + "_one_hot" for col in string_columns] + ["is_weekend"]
vector_assembler = VectorAssembler(inputCols = features, outputCol= "features")
dataset = vector_assembler.transform(df)

#
# Split the dataset into train and test
#
(training_data, test_data) = dataset.randomSplit([0.7, 0.3], 24)

print("######################################")
print("PRINTING 10 ROWS OF TRAINING DATA")
print("######################################")

training_data.show(10)

#----------------- Training phase -----------------#

#
# Initialize the classifier
#
classifier = RandomForestClassifier(labelCol = "install", featuresCol = "features", numTrees = 20)

#
# Train the model
#
classifier_model = classifier.fit(training_data)

#----------------- Testing phase -----------------#

predictions = classifier_model.transform(test_data)
evaluator= BinaryClassificationEvaluator(labelCol = "install", rawPredictionCol="probability", metricName= "areaUnderROC")
accuracy = evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy}")

with open('/usr/local/spark/app/test.txt', 'a') as the_file:
    the_file.write(f"Accuracy: {accuracy}")