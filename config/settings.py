import os
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Keep Ray's zero-GPU accelerator behavior explicit and silence its FutureWarning.
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

class SentimentSettings(BaseModel):
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    num_replicas: int = 2
    num_cpus: float = 1
    device: int = -1   # -1 CPU, 0 GPU0, etc — override via env, don't hardcode

class SummarizationSettings(BaseModel):
    model_name: str = "sshleifer/distilbart-cnn-12-6"   # small, fast default
    num_replicas: int = 1
    num_cpus: float = 2          # summarization is heavier per request
    device: int = -1
    
class Settings(BaseSettings):
    host: str = "0.0.0.0"
    http_port: int = 8000
    sentiment: SentimentSettings = SentimentSettings()
    summarization: SummarizationSettings = SummarizationSettings()
    ray_accel: bool = os.getenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO") == "1"

    class Config:
        env_prefix = "ML_"
        env_nested_delimiter = "__"

settings = Settings()