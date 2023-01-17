import mlflow
from pathlib import Path
import tempfile
import joblib

from python.src.train import train

NUM_EPOCHS = 2  # 10
f = 0.1
batch_size = 2
path = "../Data/blocksworld"
N_STEPS_MOSER = 1000

MODEL_REGISTRY = Path("experiment_tracking/experiments_storing")
EXPERIMENT_NAME = "mlflow-demo2"


def experiment_tracking_train(
    MODEL_REGISTRY,
    EXPERIMENT_NAME,
    batch_size,
    f,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    path,
    img_path=False,
    model_path=False,
):
    Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # train and evaluate
        artifacts = train(
            batch_size,
            f,
            NUM_EPOCHS,
            N_STEPS_MOSER,
            N_RUNS_MOSER,
            path,
            img_path,
            model_path,
            experiment_tracking=True,
        )
        # log key hyperparameters
        mlflow.log_params(
            {
                "f": f,
                "batch_size": batch_size,
                "NUM_EPOCHS": NUM_EPOCHS,
                "N_STEPS_MOSER": N_STEPS_MOSER,
            }
        )
        # log params which are a result of learning
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["params"], Path(dp, "params.pkl"))
            mlflow.log_artifact(dp)


if __name__ == "__main__":
    experiment_tracking_train(MODEL_REGISTRY, EXPERIMENT_NAME)
