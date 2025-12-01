"""
MLflow setup and configuration script.

This script initializes MLflow for experiment tracking, model registry,
and artifact storage.
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional
import mlflow
from mlflow.tracking import MlflowClient


class MLflowSetup:
    """Handles MLflow setup and configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MLflow setup.
        
        Args:
            config_path: Path to MLOps configuration file
        """
        self.config_path = config_path or "configs/mlops_config.yaml"
        self.config = self._load_config()
        self.client = None
        
    def _load_config(self) -> Dict:
        """Load MLOps configuration."""
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file not found at {self.config_path}")
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('mlflow', self._get_default_config())
    
    def _get_default_config(self) -> Dict:
        """Get default MLflow configuration."""
        return {
            'tracking_uri': './mlruns',
            'artifact_location': './mlartifacts',
            'backend_store_uri': 'sqlite:///mlflow.db',
            'default_artifact_root': './mlartifacts',
            'experiments': [
                {
                    'name': 'ner_training',
                    'tags': {'model_type': 'ner', 'task': 'named_entity_recognition'}
                },
                {
                    'name': 'classification_training',
                    'tags': {'model_type': 'classifier', 'task': 'multi_label_classification'}
                }
            ],
            'server': {
                'host': '0.0.0.0',
                'port': 5000,
                'workers': 2
            }
        }
    
    def setup_directories(self):
        """Create necessary directories for MLflow."""
        print("Creating MLflow directories...")
        
        directories = [
            self.config.get('tracking_uri', './mlruns'),
            self.config.get('artifact_location', './mlartifacts'),
        ]
        
        for directory in directories:
            if not directory.startswith('http'):  # Skip remote URIs
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"✓ Created {directory}")
    
    def configure_tracking(self):
        """Configure MLflow tracking."""
        print("\nConfiguring MLflow tracking...")
        
        tracking_uri = self.config.get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"✓ Tracking URI set to: {tracking_uri}")
        
        # Initialize client
        self.client = MlflowClient()
        print("✓ MLflow client initialized")
    
    def create_experiments(self):
        """Create experiments if they don't exist."""
        print("\nCreating experiments...")
        
        experiments = self.config.get('experiments', [])
        
        for exp_config in experiments:
            exp_name = exp_config['name']
            exp_tags = exp_config.get('tags', {})
            
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(exp_name)
            
            if experiment is None:
                # Create experiment
                experiment_id = mlflow.create_experiment(
                    exp_name,
                    artifact_location=self.config.get('artifact_location'),
                    tags=exp_tags
                )
                print(f"✓ Created experiment: {exp_name} (ID: {experiment_id})")
            else:
                print(f"✓ Experiment already exists: {exp_name} (ID: {experiment.experiment_id})")
    
    def setup_model_registry(self):
        """Setup model registry."""
        print("\nSetting up model registry...")
        
        # Model registry is automatically available with MLflow
        # Just verify it's accessible
        try:
            registered_models = self.client.search_registered_models()
            print(f"✓ Model registry accessible ({len(registered_models)} models registered)")
        except Exception as e:
            print(f"Warning: Could not access model registry: {e}")
    
    def test_connection(self):
        """Test MLflow connection."""
        print("\nTesting MLflow connection...")
        
        try:
            # Create a test run
            with mlflow.start_run(run_name="test_connection") as run:
                mlflow.log_param("test_param", "test_value")
                mlflow.log_metric("test_metric", 1.0)
                run_id = run.info.run_id
            
            print(f"✓ Successfully created test run: {run_id}")
            
            # Clean up test run
            self.client.delete_run(run_id)
            print("✓ Test run deleted")
            
            return True
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False
    
    def start_ui(self, background: bool = False):
        """Start MLflow UI server."""
        print("\nStarting MLflow UI...")
        
        server_config = self.config.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 5000)
        
        tracking_uri = self.config.get('tracking_uri', './mlruns')
        
        cmd = [
            'mlflow', 'ui',
            '--host', host,
            '--port', str(port),
            '--backend-store-uri', tracking_uri
        ]
        
        if background:
            print(f"Starting MLflow UI in background at http://{host}:{port}")
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✓ MLflow UI started in background")
        else:
            print(f"Starting MLflow UI at http://{host}:{port}")
            print("Press Ctrl+C to stop")
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                print("\n✓ MLflow UI stopped")
    
    def export_environment(self):
        """Export MLflow environment variables."""
        print("\nExporting environment variables...")
        
        tracking_uri = self.config.get('tracking_uri', './mlruns')
        
        env_vars = {
            'MLFLOW_TRACKING_URI': tracking_uri,
            'MLFLOW_ARTIFACT_ROOT': self.config.get('artifact_location', './mlartifacts')
        }
        
        # Print export commands
        print("\nAdd these to your environment:")
        print("=" * 50)
        for key, value in env_vars.items():
            print(f"export {key}={value}")
        print("=" * 50)
        
        # Create .env file
        env_file = Path('.env.mlflow')
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"\n✓ Environment variables saved to {env_file}")
        print(f"  Run: source {env_file}")
    
    def print_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 60)
        print("MLflow Setup Summary")
        print("=" * 60)
        print(f"Tracking URI: {self.config.get('tracking_uri')}")
        print(f"Artifact Location: {self.config.get('artifact_location')}")
        print(f"Experiments: {len(self.config.get('experiments', []))}")
        print(f"UI Host: {self.config.get('server', {}).get('host', '0.0.0.0')}")
        print(f"UI Port: {self.config.get('server', {}).get('port', 5000)}")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start MLflow UI: python scripts/setup_mlflow.py --start-ui")
        print("2. Train models: python scripts/train_ner.py --use-mlflow")
        print("3. View experiments: http://localhost:5000")
        print("=" * 60)
    
    def run_setup(self, start_ui: bool = False, background: bool = False):
        """Run complete MLflow setup."""
        print("=" * 60)
        print("MLflow Setup")
        print("=" * 60)
        
        try:
            self.setup_directories()
            self.configure_tracking()
            self.create_experiments()
            self.setup_model_registry()
            self.test_connection()
            self.export_environment()
            self.print_summary()
            
            if start_ui:
                self.start_ui(background=background)
            
            print("\n✓ MLflow setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n✗ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup MLflow for experiment tracking and model registry"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mlops_config.yaml',
        help='Path to MLOps configuration file'
    )
    parser.add_argument(
        '--start-ui',
        action='store_true',
        help='Start MLflow UI after setup'
    )
    parser.add_argument(
        '--background',
        action='store_true',
        help='Start UI in background'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only test connection without full setup'
    )
    
    args = parser.parse_args()
    
    setup = MLflowSetup(config_path=args.config)
    
    if args.test_only:
        setup.configure_tracking()
        success = setup.test_connection()
        sys.exit(0 if success else 1)
    else:
        success = setup.run_setup(
            start_ui=args.start_ui,
            background=args.background
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
