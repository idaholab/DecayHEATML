import unittest
import pandas as pd
from model_trainer import ModelTrainer

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        sample_data = pd.DataFrame({
            'time': np.random.normal(size=100),
            'power_density': np.random.uniform(1, 100, 100),
            'humidity_content': np.random.uniform(0, 0.1, 100),
            'air_ingress': np.random.uniform(0, 0.1, 100),
            'decay_heat': np.random.normal(size=100)
        })
        self.trainer = ModelTrainer(sample_data)

    def test_training(self):
        results = self.trainer.train_and_evaluate()
        self.assertGreater(len(results), 5)
        for model, metrics in results.items():
            if metrics:  # Some models might fail with small data
                self.assertIn('cv_mse', metrics)
                self.assertIn('test_r2', metrics)

if __name__ == '__main__':
    unittest.main()
