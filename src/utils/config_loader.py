"""
Configuration loader for Medical AI Assistant.
Handles loading and validating YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv


@dataclass
class NERConfig:
    """NER model configuration."""
    model_type: str = "transformer"  # "bilstm_crf" or "transformer"
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    entity_types: list = field(default_factory=lambda: ["PROBLEM", "TREATMENT", "TEST", "ANATOMY"])
    use_crf: bool = True
    dropout: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 500
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    max_entity_length: int = 20
    entity_linking: bool = True
    post_processing: bool = True
    model_save_dir: str = "models/ner"
    cache_dir: str = "cache"


@dataclass
class ClassifierConfig:
    """Clinical classifier configuration."""
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    num_labels: int = 8
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    weight_decay: float = 0.01
    warmup_steps: int = 500
    label_names: list = field(default_factory=list)
    dropout: float = 0.1
    hidden_dim: int = 768
    model_save_dir: str = "models/classifier"
    cache_dir: str = "cache"


@dataclass
class ASRConfig:
    """ASR model configuration."""
    model_name: str = "openai/whisper-medium"
    language: str = "en"
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 5
    max_audio_length: int = 30  # seconds
    use_medical_vocabulary: bool = True


@dataclass
class RAGConfig:
    """RAG system configuration."""
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768
    vector_db: str = "faiss"  # "faiss" or "chromadb"
    vector_db_path: str = "data/vector_db"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_doc: int = 100
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_hybrid_search: bool = True
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10
    use_query_expansion: bool = True
    use_query_rewriting: bool = True
    max_query_length: int = 256
    max_context_length: int = 2048
    include_sources: bool = True
    min_confidence: float = 0.5
    medical_literature_path: str = "data/medical_literature"
    cache_dir: str = "cache"


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"  # "openai" or "anthropic"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    use_prompt_templates: bool = True
    system_prompt: str = ""
    enable_content_filtering: bool = True
    max_retries: int = 3
    timeout: int = 30
    use_cache: bool = True
    cache_dir: str = "cache/llm"


@dataclass
class SafetyConfig:
    """Safety guardrails configuration."""
    enable_emergency_detection: bool = True
    emergency_keywords: list = field(default_factory=list)
    enable_claim_filtering: bool = True
    prohibited_claims: list = field(default_factory=list)
    enable_dosage_validation: bool = True
    max_dosage_threshold: float = 10.0
    min_dosage_threshold: float = 0.1
    confidence_threshold: float = 0.7
    low_confidence_message: str = ""
    add_disclaimers: bool = True
    disclaimer_text: str = ""
    emergency_disclaimer: str = ""
    log_safety_events: bool = True
    safety_log_path: str = "logs/safety_events.log"


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    log_level: str = "info"
    enable_auth: bool = True
    enable_rate_limit: bool = True
    rate_limit_per_minute: int = 60
    enable_metrics: bool = True
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    cors: Optional[Dict] = None
    rate_limit: Optional[Dict] = None
    request_limits: Optional[Dict] = None
    models: Optional[Dict] = None
    monitoring: Optional[Dict] = None
    logging: Optional[Dict] = None
    authentication: Optional[Dict] = None
    docs: Optional[Dict] = None
    cors_enabled: bool = True
    cors_origins: Optional[list] = None
    refresh_token_expire_days: int = 30
    rate_limit_per_hour: int = 1000
    metrics_port: int = 9090
    enable_tracing: bool = False
    api_version: str = "v1"
    api_prefix: str = "/api"
    max_request_size: int = 10485760
    request_timeout: int = 30
    health_check_interval: int = 30
    static_dir: str = "static"
    upload_dir: str = "uploads"


@dataclass
class MLOpsConfig:
    """MLOps configuration."""
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "medical-ai-assistant"
    mlflow_artifact_location: str = "mlruns"
    use_dvc: bool = True
    dvc_remote: str = "local"
    dvc_cache_dir: str = ".dvc/cache"
    use_wandb: bool = False
    wandb_project: str = "medical-ai-assistant"
    wandb_entity: str = ""
    model_registry: str = "mlflow"
    model_stage: str = "staging"
    log_metrics: bool = True
    log_parameters: bool = True
    log_artifacts: bool = True
    log_model: bool = True
    enable_monitoring: bool = True
    monitoring_interval: int = 60
    alert_on_drift: bool = True
    drift_threshold: float = 0.1
    enable_ab_testing: bool = False
    ab_test_config: Optional[Dict] = None
    models_dir: str = "models"
    experiments_dir: str = "experiments"
    artifacts_dir: str = "artifacts"


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)
            
        # Load environment variables
        load_dotenv()
    
    def load_yaml(self, config_file: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            config_file: Name of configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # ✅ FIX: Add encoding='utf-8' here
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    def load_ner_config(self) -> NERConfig:
        """Load NER configuration."""
        try:
            config = self.load_yaml("ner_config.yaml")
            return NERConfig(**config)
        except FileNotFoundError:
            return NERConfig()
    
    def load_classifier_config(self) -> ClassifierConfig:
        """Load classifier configuration."""
        try:
            config = self.load_yaml("classifier_config.yaml")
            return ClassifierConfig(**config)
        except FileNotFoundError:
            return ClassifierConfig()
    
    def load_asr_config(self) -> ASRConfig:
        """Load ASR configuration."""
        try:
            config = self.load_yaml("asr_config.yaml")
            return ASRConfig(**config)
        except FileNotFoundError:
            return ASRConfig()
    
    def load_rag_config(self) -> RAGConfig:
        """Load RAG configuration."""
        try:
            config = self.load_yaml("rag_config.yaml")
            return RAGConfig(**config)
        except FileNotFoundError:
            return RAGConfig()
    
    def load_llm_config(self) -> LLMConfig:
        """Load LLM configuration."""
        try:
            config = self.load_yaml("llm_config.yaml")
            return LLMConfig(**config)
        except FileNotFoundError:
            return LLMConfig()
    
    def load_safety_config(self) -> SafetyConfig:
        """Load safety configuration."""
        try:
            config = self.load_yaml("safety_config.yaml")
            return SafetyConfig(**config)
        except FileNotFoundError:
            return SafetyConfig()
    
    def load_api_config(self) -> APIConfig:
        """Load API configuration."""
        try:
            config = self.load_yaml("api_config.yaml")
            return APIConfig(**config)
        except FileNotFoundError:
            return APIConfig()
    
    def load_mlops_config(self) -> MLOpsConfig:
        """Load MLOps configuration."""
        try:
            config = self.load_yaml("mlops_config.yaml")
            return MLOpsConfig(**config)
        except FileNotFoundError:
            return MLOpsConfig()
    
    def get_env_variable(self, key: str, default: Optional[str] = None) -> str:
        """
        Get environment variable.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
