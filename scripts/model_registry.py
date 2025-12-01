"""
Model registry utilities for managing trained models.

This module provides utilities for:
- Model versioning and registration
- Model promotion (staging -> production)
- Model deployment utilities
- Model metadata management
"""

import os
import sys
import json
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion


class ModelRegistry:
    """Manages model registration and deployment."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        self.tracking_uri = mlflow.get_tracking_uri()
    
    def register_model(
        self,
        model_path: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict] = None,
        run_id: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a model in MLflow model registry.
        
        Args:
            model_path: Path to model artifacts
            model_name: Name for the registered model
            description: Model description
            tags: Model tags
            run_id: MLflow run ID
            
        Returns:
            ModelVersion object
        """
        print(f"Registering model: {model_name}")
        
        # Register model
        if run_id:
            model_uri = f"runs:/{run_id}/{model_path}"
        else:
            model_uri = model_path
        
        try:
            # Register the model
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            print(f"✓ Model registered: {model_name} (version {result.version})")
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=result.version,
                    description=description
                )
                print(f"✓ Description updated")
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=result.version,
                        key=key,
                        value=value
                    )
                print(f"✓ Tags added: {tags}")
            
            return result
            
        except Exception as e:
            print(f"✗ Failed to register model: {e}")
            raise
    
    def promote_model(
        self,
        model_name: str,
        version: Union[int, str],
        stage: str
    ):
        """
        Promote model to a specific stage.
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        print(f"Promoting {model_name} v{version} to {stage}...")
        
        valid_stages = ["Staging", "Production", "Archived", "None"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of {valid_stages}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"✓ Model promoted to {stage}")
            
        except Exception as e:
            print(f"✗ Failed to promote model: {e}")
            raise
    
    def get_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None
    ) -> List[ModelVersion]:
        """
        Get model versions.
        
        Args:
            model_name: Registered model name
            stages: Filter by stages (e.g., ["Production", "Staging"])
            
        Returns:
            List of ModelVersion objects
        """
        try:
            if stages:
                versions = []
                for stage in stages:
                    stage_versions = self.client.get_latest_versions(
                        name=model_name,
                        stages=[stage]
                    )
                    versions.extend(stage_versions)
            else:
                versions = self.client.search_model_versions(
                    f"name='{model_name}'"
                )
            
            return versions
            
        except Exception as e:
            print(f"Warning: Could not get model versions: {e}")
            return []
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get production model version.
        
        Args:
            model_name: Registered model name
            
        Returns:
            ModelVersion object or None
        """
        versions = self.get_model_versions(model_name, stages=["Production"])
        return versions[0] if versions else None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[Union[int, str]] = None,
        stage: Optional[str] = None
    ):
        """
        Load a registered model.
        
        Args:
            model_name: Registered model name
            version: Model version (if not using stage)
            stage: Model stage (if not using version)
            
        Returns:
            Loaded model
        """
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        elif version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            # Load latest production model
            model_uri = f"models:/{model_name}/Production"
        
        print(f"Loading model: {model_uri}")
        
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✓ Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def compare_models(
        self,
        model_name: str,
        version1: Union[int, str],
        version2: Union[int, str]
    ) -> Dict:
        """
        Compare two model versions.
        
        Args:
            model_name: Registered model name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dictionary
        """
        print(f"Comparing {model_name} v{version1} vs v{version2}...")
        
        try:
            # Get model versions
            mv1 = self.client.get_model_version(model_name, version1)
            mv2 = self.client.get_model_version(model_name, version2)
            
            # Get run details
            run1 = self.client.get_run(mv1.run_id)
            run2 = self.client.get_run(mv2.run_id)
            
            comparison = {
                'version1': {
                    'version': version1,
                    'stage': mv1.current_stage,
                    'metrics': run1.data.metrics,
                    'params': run1.data.params,
                    'created': mv1.creation_timestamp
                },
                'version2': {
                    'version': version2,
                    'stage': mv2.current_stage,
                    'metrics': run2.data.metrics,
                    'params': run2.data.params,
                    'created': mv2.creation_timestamp
                }
            }
            
            # Print comparison
            self._print_comparison(comparison)
            
            return comparison
            
        except Exception as e:
            print(f"✗ Failed to compare models: {e}")
            raise
    
    def _print_comparison(self, comparison: Dict):
        """Print model comparison."""
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        
        v1 = comparison['version1']
        v2 = comparison['version2']
        
        print(f"\nVersion {v1['version']} (Stage: {v1['stage']})")
        print("Metrics:")
        for metric, value in v1['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nVersion {v2['version']} (Stage: {v2['stage']})")
        print("Metrics:")
        for metric, value in v2['metrics'].items():
            v1_value = v1['metrics'].get(metric)
            if v1_value:
                diff = value - v1_value
                sign = "+" if diff > 0 else ""
                print(f"  {metric}: {value:.4f} ({sign}{diff:.4f})")
            else:
                print(f"  {metric}: {value:.4f}")
        
        print("=" * 60)
    
    def archive_old_models(
        self,
        model_name: str,
        keep_latest: int = 3
    ):
        """
        Archive old model versions.
        
        Args:
            model_name: Registered model name
            keep_latest: Number of latest versions to keep
        """
        print(f"Archiving old versions of {model_name}...")
        
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Sort by version number
            versions = sorted(
                versions,
                key=lambda x: int(x.version),
                reverse=True
            )
            
            # Archive old versions
            archived = 0
            for version in versions[keep_latest:]:
                if version.current_stage not in ["Production", "Archived"]:
                    self.promote_model(
                        model_name,
                        version.version,
                        "Archived"
                    )
                    archived += 1
            
            print(f"✓ Archived {archived} old versions")
            
        except Exception as e:
            print(f"Warning: Could not archive models: {e}")
    
    def export_model(
        self,
        model_name: str,
        version: Union[int, str],
        output_dir: str
    ):
        """
        Export model to local directory.
        
        Args:
            model_name: Registered model name
            version: Model version
            output_dir: Output directory
        """
        print(f"Exporting {model_name} v{version}...")
        
        try:
            # Get model version
            mv = self.client.get_model_version(model_name, version)
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Download artifacts
            artifact_path = self.client.download_artifacts(
                mv.run_id,
                "model",
                str(output_path)
            )
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'version': version,
                'stage': mv.current_stage,
                'run_id': mv.run_id,
                'created': mv.creation_timestamp,
                'description': mv.description
            }
            
            metadata_path = output_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Model exported to {output_path}")
            
        except Exception as e:
            print(f"✗ Failed to export model: {e}")
            raise
    
    def list_registered_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        try:
            models = self.client.search_registered_models()
            model_names = [model.name for model in models]
            
            print("Registered Models:")
            for name in model_names:
                print(f"  - {name}")
            
            return model_names
            
        except Exception as e:
            print(f"Warning: Could not list models: {e}")
            return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Model registry utilities"
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Register model
    register_parser = subparsers.add_parser('register', help='Register a model')
    register_parser.add_argument('--model-path', required=True, help='Model path')
    register_parser.add_argument('--name', required=True, help='Model name')
    register_parser.add_argument('--description', help='Model description')
    register_parser.add_argument('--run-id', help='MLflow run ID')
    
    # Promote model
    promote_parser = subparsers.add_parser('promote', help='Promote model')
    promote_parser.add_argument('--name', required=True, help='Model name')
    promote_parser.add_argument('--version', required=True, help='Model version')
    promote_parser.add_argument('--stage', required=True, help='Target stage')
    
    # Compare models
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('--name', required=True, help='Model name')
    compare_parser.add_argument('--version1', required=True, help='First version')
    compare_parser.add_argument('--version2', required=True, help='Second version')
    
    # Export model
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--name', required=True, help='Model name')
    export_parser.add_argument('--version', required=True, help='Model version')
    export_parser.add_argument('--output', required=True, help='Output directory')
    
    # List models
    subparsers.add_parser('list', help='List registered models')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize registry
    registry = ModelRegistry(tracking_uri=args.tracking_uri)
    
    # Execute command
    try:
        if args.command == 'register':
            registry.register_model(
                model_path=args.model_path,
                model_name=args.name,
                description=args.description,
                run_id=args.run_id
            )
        elif args.command == 'promote':
            registry.promote_model(
                model_name=args.name,
                version=args.version,
                stage=args.stage
            )
        elif args.command == 'compare':
            registry.compare_models(
                model_name=args.name,
                version1=args.version1,
                version2=args.version2
            )
        elif args.command == 'export':
            registry.export_model(
                model_name=args.name,
                version=args.version,
                output_dir=args.output
            )
        elif args.command == 'list':
            registry.list_registered_models()
        
        print("\n✓ Command completed successfully")
        
    except Exception as e:
        print(f"\n✗ Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
