"""
Train a DGA detector with H2O AutoML and export the leader as a MOJO.

Inputs:
- hands-on/dga_dataset_train.csv (created by the provided data generator)

Outputs:
- model/DGA_Leader.zip (MOJO for production scoring)
- hands-on/models/best_dga_model (H2O binary model for local scoring)
"""

import os
from pathlib import Path

import h2o
from h2o.automl import H2OAutoML


def find_dataset_path() -> str:
    candidate_paths = [
        "hands-on/dga_dataset_train.csv",
        "dga_dataset_train.csv",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find dga_dataset_train.csv. Generate it with hands-on/1_generate_dga_data.py"
    )


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    dataset_path = find_dataset_path()

    h2o.init()

    train = h2o.import_file(dataset_path)
    features = ["length", "entropy"]
    target = "class"
    train[target] = train[target].asfactor()

    automl = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1)
    automl.train(x=features, y=target, training_frame=train)

    print("H2O AutoML complete. Leaderboard head:")
    print(automl.leaderboard.head())

    leader = automl.leader

    ensure_dir("model")
    ensure_dir("hands-on/models")

    mojo_path = leader.download_mojo(path="model")
    print(f"MOJO saved: {mojo_path}")

    desired_mojo = os.path.join("model", "DGA_Leader.zip")
    try:
        if os.path.abspath(mojo_path) != os.path.abspath(desired_mojo):
            # Overwrite if exists to keep name stable
            if os.path.exists(desired_mojo):
                os.remove(desired_mojo)
            os.replace(mojo_path, desired_mojo)
            mojo_path = desired_mojo
    except Exception as e:
        print(f"Warning: Could not rename MOJO to {desired_mojo}: {e}")

    saved_model_path = h2o.save_model(model=leader, path="hands-on/models", force=True)
    stable_binary_dir = os.path.join("hands-on", "models", "best_dga_model")
    try:
        if os.path.abspath(saved_model_path) != os.path.abspath(stable_binary_dir):
            if os.path.exists(stable_binary_dir):
                # Remove previous directory/file
                if os.path.isdir(stable_binary_dir):
                    import shutil

                    shutil.rmtree(stable_binary_dir)
                else:
                    os.remove(stable_binary_dir)
            os.replace(saved_model_path, stable_binary_dir)
            saved_model_path = stable_binary_dir
    except Exception as e:
        print(f"Warning: Could not stabilize binary model path {stable_binary_dir}: {e}")

    print("---")
    print(f"Leader MOJO: {mojo_path}")
    print(f"Binary model: {saved_model_path}")

    h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()


