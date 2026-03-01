
from pydantic import BaseSettings

class Settings(BaseSettings):
   
    APP_NAME : str = "HealthBot"
    APP_VERSION : str = "0.1.0"
    DEBUG : bool = True

    APP_HOST : str = "0.0.0.0"
    APP_PORT : int = 8000
        