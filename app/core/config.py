import os
from typing import List

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Simplex FICCT - Two Phase Method")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    
    # Orígenes permitidos (configurable por variable de entorno separada por comas)
    _cors_env: str = os.getenv("BACKEND_CORS_ORIGINS", "")
    BACKEND_CORS_ORIGINS: List[str] = (
        [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
        if _cors_env
        else [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://metodo-simplex-frontend.vercel.app"
        ]
    )
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
