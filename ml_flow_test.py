import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[NL] [Amsterdam] [MartonMunkacsi] TaxiFareModel + 1.0.0"

# Indicate mlflow to log to remote server
mlflow.set_tracking_uri(MLFLOW_URI)

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(
        EXPERIMENT_NAME).experiment_id

yourname = "Marton"

if yourname is None:
    print("please define your name, it will be used as a parameter to log")

for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 4.5)
    client.log_param(run.info.run_id, "model", model)
    client.log_param(run.info.run_id, "student_name", yourname)
