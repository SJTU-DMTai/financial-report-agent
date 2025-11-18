import yaml

class Config:
    def __init__(self, path: str = "config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def get_model_cfg(self):
        models = self.data["models"]
        model_id = models["default"]
        return models[model_id]
