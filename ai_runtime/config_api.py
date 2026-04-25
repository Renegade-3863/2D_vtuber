import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class APIConfig:
    # LLM_PROVIDER: "qwen" (默认, 复用 DashScope key) | "azure" (旧的 hkust gpt-4o-mini)
    provider: str = os.getenv("LLM_PROVIDER", "qwen").lower()

    # ---- Qwen / DashScope (OpenAI 兼容模式) ----
    qwen_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    qwen_endpoint: str = os.getenv(
        "QWEN_ENDPOINT",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # qwen-plus: 性价比首选; qwen-max: 最强但贵; qwen-turbo: 便宜
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-plus")

    # ---- Azure (legacy / fallback) ----
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT", "https://hkust.azure-api.net/")
    azure_deployment: str = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2025-02-01-preview")

    timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # 统一对外接口：根据 provider 暴露同名字段，老代码不用改
    @property
    def api_key(self) -> str:
        return self.qwen_api_key if self.provider == "qwen" else self.azure_api_key

    @property
    def endpoint(self) -> str:
        return self.qwen_endpoint if self.provider == "qwen" else self.azure_endpoint

    @property
    def deployment(self) -> str:
        return self.qwen_model if self.provider == "qwen" else self.azure_deployment

    @property
    def api_version(self) -> str:
        return self.azure_api_version  # qwen 不需要

cfg = APIConfig()

print(f"Loaded LLM provider={cfg.provider}, model={cfg.deployment}, endpoint={cfg.endpoint}")