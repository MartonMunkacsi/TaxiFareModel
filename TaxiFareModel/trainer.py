# imports
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
# from ml_flow_test import MLFLOW_URI
import joblib
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', Lasso())])
        return pipe

    def run(self):
        """set and train the pipeline"""
        # Fit
        pipe = self.set_pipeline()
        pipeline = pipe.fit(self.X, self.y)
        return pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipeline = self.run()
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    data = get_data()

    data = clean_data(data)

    X = data.drop(columns='fare_amount')
    y = data['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    my_class = Trainer(X_train, y_train)

    my_class.run()
    evaluate_data = my_class.evaluate(X_test, y_test)
    my_class.save_model()
    print(evaluate_data)

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[NL] [Amsterdam] [MartonMunkacsi] TaxiFareModel + 1.0.0"
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
        client.log_metric(run.info.run_id, "rmse", evaluate_data)
        client.log_param(run.info.run_id, "model", model)
        client.log_param(run.info.run_id, "student_name", yourname)
