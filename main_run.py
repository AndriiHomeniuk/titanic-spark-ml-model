from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def update_dataframe(main_df):
    main_dataframe = main_df.select(
        col('Survived').cast('float'),
        col('Pclass').cast('float'),
        col('Sex'),
        col('Age').cast('float'),
        col('Fare').cast('float'),
        col('Embarked')
    ).replace('?', None).dropna(how='any')

    # Add new column with coded info
    main_dataframe = StringIndexer(
        inputCol='Sex',
        outputCol='Gender',
        handleInvalid='keep'
    ).fit(main_dataframe).transform(main_dataframe)
    main_dataframe = StringIndexer(
        inputCol='Embarked',
        outputCol='Boarded',
        handleInvalid='keep'
    ).fit(main_dataframe).transform(main_dataframe)

    return main_dataframe.drop('Sex').drop('Embarked')


def transform_data(input_cols, raw_dataset):
    # Assemble all the features with VectorAssembler
    return VectorAssembler(inputCols=input_cols, outputCol='features').transform(raw_dataset)


if __name__ == '__main__':
    spark = SparkSession.builder\
        .appName('Titanic Data')\
        .getOrCreate()

    df = (spark.read
          .format('csv')
          .option('header', 'true')
          .load('train.csv'))
    dataset = update_dataframe(df)

    required_features = [
        'Pclass',
        'Age',
        'Fare',
        'Gender',
        'Boarded',
    ]
    transformed_data = transform_data(required_features, dataset)
    # Creating test and train data
    (training_data, test_data) = transformed_data.randomSplit([0.8, 0.2])

    # Make model and predictions
    rf = RandomForestClassifier(
        labelCol='Survived',
        featuresCol='features',
        maxDepth=5,
    )
    model = rf.fit(training_data)
    predictions = model.transform(test_data)

    # Evaluate model
    evaluator = MulticlassClassificationEvaluator(
        labelCol='Survived',
        predictionCol='prediction',
        metricName='accuracy',
    )
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy = ', accuracy)
