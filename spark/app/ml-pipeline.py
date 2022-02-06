import sys
from pyspark.sql import SparkSession, functions
from pyspark.sql.types import StringType
from modules import moduleExample
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, dayofweek, datediff, hour, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from collections import defaultdict
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd

def undersampleDataset(df):
    
    majorClass = df.filter(col("install") == 0)
    minorClass = df.filter(col("install") == 1)

    ratio = int(majorClass.count() / minorClass.count())
    sampledMajorClass = majorClass.sample(False, 1 / ratio)

    return sampledMajorClass.unionAll(minorClass)

def partOfTheDay(hour):
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

def processDates(df):

    #
    # Get day of the week as string
    #
    df = df.withColumn("day", date_format('timestamp', 'E').alias('dow_string'))

    #
    # Determine if day is during the weekend
    #
    df = df.withColumn("is_weekend", dayofweek("timestamp").isin([1,7]).cast("int"))

    #
    # Compute the difference LastStart and Timestamp
    #
    df = df.withColumn("diff_last_start", datediff(df.timestamp, df.lastStart))

    #
    # Keep the hour the entry was recorded at
    #
    df = df.withColumn("hour", hour('timestamp').alias('hour'))

    #
    # Determine part of the day the entry was recorded at
    #
    udfPartOfDay = udf(partOfTheDay, StringType())
    df = df.withColumn('partOfDay', udfPartOfDay(df.hour))

    return df

def castColumnsToDouble(df, columnsToCast):

    #
    # Cast given colums to double
    #
    castedColumns = []
    for col in columnsToCast:
        try:
            df = df.withColumn(col + "_double", df[col].cast("double"))
            castedColumns.append(col + "_double")
        except Exception as e:
            print(f"ERROR: {col} could not be casted to double.")
            print(e)

    return df, castedColumns

def processNumericalFeatures(df, castedColumns):
    #
    # Prepare imputed columns names
    #
    imputedColumns = [col + "_imputed" for col in castedColumns]

    #
    # Completes missing values in the dataset, handle NaN
    #
    imputerStage = Imputer(inputCols = castedColumns, outputCols = imputedColumns)

    #
    # Prepare the data for StandardScaler
    #
    assemblerStage = VectorAssembler(inputCols = imputedColumns, outputCol = "features_vector_assembled")

    #
    # Normalize each feature to have unit standard deviation 
    #
    scalerStage = StandardScaler(inputCol = "features_vector_assembled", outputCol = "features_scaled", withStd = True, withMean = False)

    #
    # Run the stages
    #
    pipeline = Pipeline(stages = [imputerStage, assemblerStage, scalerStage])
    df = pipeline.fit(df).transform(df)

    return df

def determineTypes(df):
    #
    # Determine type of all columns
    #
    columnsTypes = defaultdict(list)
    for entry in df.schema.fields:
        columnsTypes[str(entry.dataType)].append(entry.name)

    return columnsTypes

def processCategoricalFeatures(df):
    #
    # Get all string/categorical feature names
    #
    columnsTypes = determineTypes(df)
    categoricalFeatures = [col for col in columnsTypes["StringType"]]

    #
    # Prepare features for OneHotEncoder
    #
    stringIndexerStage = [StringIndexer(inputCol = col, outputCol = col + "_string_encoded") for col in categoricalFeatures]

    #
    # Apply OneHotEncoding to categorical feature
    #
    oneHotEncoderStage = [OneHotEncoder(inputCol = col + "_string_encoded", outputCol = col + "_one_hot") for col in categoricalFeatures]

    #
    # Run the stages
    #
    pipeline = Pipeline(stages = stringIndexerStage + oneHotEncoderStage)
    df = pipeline.fit(df).transform(df)

    return df, categoricalFeatures

def logDebugMessage(message):
    print("######################################")
    print(message)
    print("######################################")

def logLoss(predictions, testData):
    #
    # Series of cast to prepare for evaluation
    #

    # select data
    Y = testData.select('install')
    probabilities = predictions.select('probability')

    # cast spark dataframe -> numpy array
    probabilities = np.array(predictions.select('probability').collect())
    probabilities = probabilities.reshape(-1, probabilities.shape[-1])

    # cast spark dataframe -> pandas dataframe -> panda series
    Y = Y.toPandas()
    Y = pd.Series(Y['install'].values)
    
    return log_loss(Y, probabilities)

def main():

    #
    # Get start parameters
    #
    logDebug = bool(sys.argv[1])
    csvFileName = sys.argv[2]

    #
    # Initialize Spark
    #
    if logDebug:
        logDebugMessage("Initializing Spark")

    spark = (SparkSession
        .builder
        .getOrCreate()
    )
    sc = spark.sparkContext

    sc.setLogLevel("WARN")

    

    #----------------- Preprocessing phase -----------------#
    
    #
    # Load dataset
    #
    if logDebug:
        logDebugMessage("Loading dataset...")

    df = spark.read.option("delimiter", ";").option("inferschema", "true").csv(csvFileName, header = True)

    if logDebug:
        logDebugMessage("Dataset successfully loaded")

    if logDebug:
        logDebugMessage("Preprocessing data...")

    #
    # Dataset is imbalanced, let's undersample it
    #
    df = undersampleDataset(df)

    #
    # Process Dates
    #
    df = processDates(df)

    #
    # Cast desired columns to double
    #
    df, castedColumns = castColumnsToDouble(df, ["startCount", "viewCount", "clickCount", "installCount", "startCount1d", "startCount7d", "diff_last_start"])

    #
    # Impute and scale numerical features
    #
    df = processNumericalFeatures(df, castedColumns)
    
    #
    # Drop columns that are no longer useful or categorical columns with too much different values 
    #
    columns = ["id", "timestamp", "lastStart", "campaignId", "sourceGameId", "softwareVersion", "deviceType", "hour"]
    df = df.drop(*columns)
    
    #
    # OneHotEncoding categorical features
    #
    df, categoricalFeatures = processCategoricalFeatures(df)

    #
    # Create final dataset
    #
    features = ["features_scaled"] + [col + "_one_hot" for col in categoricalFeatures] + ["is_weekend"]
    vector_assembler = VectorAssembler(inputCols = features, outputCol= "features")
    dataset = vector_assembler.transform(df)

    if logDebug:
        logDebugMessage("Data preprocessed")
        dataset.show(10)


    #----------------- Training phase -----------------#

    #
    # Split the dataset into train and test
    #
    (trainingData, testData) = dataset.randomSplit([0.8, 0.2], 24)

    
    #
    # Train the model
    #
    if logDebug:
        logDebugMessage("Training the Logistic Regression model...")    

    model = LogisticRegression(featuresCol = 'features', labelCol = 'install', maxIter=10).fit(trainingData)

    #----------------- Testing phase -----------------#

    predictions = model.transform(testData)
    evaluator = BinaryClassificationEvaluator(labelCol = "install", rawPredictionCol="probability", metricName= "areaUnderROC")

    if logDebug:
        logDebugMessage("Testing the model...")

    AUC = evaluator.evaluate(predictions)
    print("Logistic Regression Performance Measure")
    print(f"AUC = {AUC}")

    loss = logLoss(predictions, testData)

    print(f"Log Loss: {loss}")


if __name__ == "__main__":
    main()