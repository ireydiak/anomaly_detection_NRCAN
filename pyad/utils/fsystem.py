import os

def get_default_experiment_path():
    return os.path.join(
        os.path.abspath(__file__), "../experiments/training"
    )