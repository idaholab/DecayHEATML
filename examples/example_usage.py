from data_loader import DataLoader
from model_trainer import ModelTrainer
from visualizer import Visualizer
from config import Config

# Load data
loader = DataLoader(Config.DATA_PATHS['cl_based'])
cases, df = loader.load_and_preprocess(Config.DEFAULT_NORMALIZATION)

# Train models
trainer = ModelTrainer(df)
results = trainer.train_and_evaluate()

# Visualize results
viz = Visualizer(trainer.models, df)
viz.create_interactive_plot()
