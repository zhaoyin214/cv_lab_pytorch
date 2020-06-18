from typing import Dict

from .common import BaseTrainer

class LandmarkTrainer(BaseTrainer):
    """landmark regression model
    """
    def _fetch_input_label(self, samples: Dict):
        return samples["image"], samples["landmarks"]

