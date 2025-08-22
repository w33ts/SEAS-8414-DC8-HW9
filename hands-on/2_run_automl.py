# Filename: 2_run_automl.py
import os
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train = h2o.import_file("dga_dataset_train.csv")
x = ['length', 'entropy'] # Features
y = "class"               # Target
train[y] = train[y].asfactor()

aml = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1)
aml.train(x=x, y=y, training_frame=train)

print("H2O AutoML process complete.")
print("Leaderboard:")
print(aml.leaderboard.head())

# (Add this to the end of 2_run_automl.py)
# Get the best performing model from the leaderboard
best_model = aml.leader

# Download the MOJO artifact.
mojo_path = best_model.download_mojo(path="./models/")


# After AutoML finishes
custom_name = "best_dga_model"
model_path = h2o.save_model(model=aml.leader, path="./models", force=True)
print(f"Original saved model path: {model_path}")

# Rename the folder to a friendly name
new_path = os.path.join("./models", custom_name)
os.rename(model_path, new_path)
print(f"Renamed model path: {new_path}")

print(f"Production-ready model saved to: {mojo_path}")
h2o.shutdown()