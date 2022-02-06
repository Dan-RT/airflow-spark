from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from datetime import datetime, timedelta

###############################################
# Parameters
###############################################
spark_master = "spark://spark:7077"
csv_file = "/usr/local/spark/resources/data/training_data.csv"

###############################################
# DAG Definition
###############################################
now = datetime.now()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(now.year, now.month, now.day),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1)
}

dag = DAG(
        dag_id="ml-pipeline", 
        description="This DAG runs a Pyspark app that uses modules.",
        default_args=default_args, 
        schedule_interval=timedelta(1)
    )

start = DummyOperator(task_id="start", dag=dag)

ML_Pipeline = SparkSubmitOperator(
    task_id="ML_Pipeline",
    application="/usr/local/spark/app/ml-pipeline.py", 
    name="ml-pipeline",
    conn_id="spark_default",
    verbose=1,
    conf={"spark.master":spark_master},
    application_args=["True", csv_file],
    dag=dag)

Model_Versioning = DummyOperator(task_id="Model_Versioning", dag=dag)

start >> ML_Pipeline >> Model_Versioning