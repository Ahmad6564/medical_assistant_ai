"""
DVC (Data Version Control) setup and configuration script.

This script initializes DVC for data versioning, pipeline management,
and remote storage configuration.
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional, List


class DVCSetup:
    """Handles DVC setup and configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DVC setup.
        
        Args:
            config_path: Path to MLOps configuration file
        """
        self.config_path = config_path or "configs/mlops_config.yaml"
        self.config = self._load_config()
        self.project_root = Path.cwd()
        
    def _load_config(self) -> Dict:
        """Load DVC configuration."""
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file not found at {self.config_path}")
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('dvc', self._get_default_config())
    
    def _get_default_config(self) -> Dict:
        """Get default DVC configuration."""
        return {
            'remotes': [
                {
                    'name': 'local',
                    'url': './dvc-storage',
                    'default': True
                }
            ],
            'cache': {
                'dir': '.dvc/cache',
                'type': 'copy'
            },
            'data_dirs': [
                'data/raw',
                'data/processed',
                'data/medical_literature'
            ],
            'model_dirs': [
                'models'
            ]
        }
    
    def check_dvc_installed(self) -> bool:
        """Check if DVC is installed."""
        try:
            result = subprocess.run(
                ['dvc', '--version'],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip()
            print(f"✓ DVC is installed: {version}")
            return True
        except FileNotFoundError:
            print("✗ DVC is not installed")
            print("  Install with: pip install dvc")
            return False
    
    def check_git_initialized(self) -> bool:
        """Check if git is initialized."""
        git_dir = self.project_root / '.git'
        if git_dir.exists():
            print("✓ Git repository initialized")
            return True
        else:
            print("✗ Git repository not initialized")
            print("  Run: git init")
            return False
    
    def initialize_dvc(self):
        """Initialize DVC in the project."""
        print("\nInitializing DVC...")
        
        dvc_dir = self.project_root / '.dvc'
        if dvc_dir.exists():
            print("✓ DVC already initialized")
            return
        
        try:
            subprocess.run(
                ['dvc', 'init'],
                check=True,
                cwd=self.project_root
            )
            print("✓ DVC initialized successfully")
            
            # Commit DVC files to git
            subprocess.run(
                ['git', 'add', '.dvc', '.dvcignore'],
                cwd=self.project_root
            )
            print("✓ DVC files staged for git commit")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to initialize DVC: {e}")
            raise
    
    def configure_remotes(self):
        """Configure DVC remotes."""
        print("\nConfiguring DVC remotes...")
        
        remotes = self.config.get('remotes', [])
        
        for remote in remotes:
            name = remote['name']
            url = remote['url']
            is_default = remote.get('default', False)
            
            # Create remote directory if local
            if not url.startswith(('s3://', 'gs://', 'azure://', 'http')):
                Path(url).mkdir(parents=True, exist_ok=True)
            
            # Add remote
            try:
                subprocess.run(
                    ['dvc', 'remote', 'add', '-f', name, url],
                    check=True,
                    cwd=self.project_root,
                    capture_output=True
                )
                print(f"✓ Added remote: {name} -> {url}")
                
                # Set as default if specified
                if is_default:
                    subprocess.run(
                        ['dvc', 'remote', 'default', name],
                        check=True,
                        cwd=self.project_root,
                        capture_output=True
                    )
                    print(f"✓ Set {name} as default remote")
                    
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not configure remote {name}: {e}")
    
    def setup_data_tracking(self):
        """Setup data tracking with DVC."""
        print("\nSetting up data tracking...")
        
        data_dirs = self.config.get('data_dirs', [])
        
        for data_dir in data_dirs:
            dir_path = self.project_root / data_dir
            
            # Create directory if it doesn't exist
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Check if already tracked
            dvc_file = Path(str(dir_path) + '.dvc')
            if dvc_file.exists():
                print(f"✓ Already tracking: {data_dir}")
                continue
            
            # Add to DVC
            try:
                subprocess.run(
                    ['dvc', 'add', data_dir],
                    check=True,
                    cwd=self.project_root,
                    capture_output=True
                )
                print(f"✓ Added to DVC: {data_dir}")
                
                # Stage .dvc file for git
                subprocess.run(
                    ['git', 'add', f"{data_dir}.dvc", '.gitignore'],
                    cwd=self.project_root
                )
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not track {data_dir}: {e}")
    
    def create_pipeline(self):
        """Create DVC pipeline configuration."""
        print("\nCreating DVC pipeline...")
        
        pipeline_config = {
            'stages': {
                'prepare_data': {
                    'cmd': 'python scripts/prepare_data.py',
                    'deps': ['data/raw'],
                    'outs': ['data/processed']
                },
                'train_ner': {
                    'cmd': 'python scripts/train_ner.py --data-path data/processed --save-dir models/ner',
                    'deps': ['data/processed', 'scripts/train_ner.py'],
                    'params': ['configs/ner_config.yaml'],
                    'outs': ['models/ner'],
                    'metrics': ['models/ner/metrics.json']
                },
                'train_classifier': {
                    'cmd': 'python scripts/train_classifier.py --data-path data/processed --save-dir models/classifier',
                    'deps': ['data/processed', 'scripts/train_classifier.py'],
                    'params': ['configs/classifier_config.yaml'],
                    'outs': ['models/classifier'],
                    'metrics': ['models/classifier/metrics.json']
                },
                'evaluate': {
                    'cmd': 'python scripts/evaluate_models.py',
                    'deps': ['models/ner', 'models/classifier', 'data/processed'],
                    'metrics': ['evaluation/results.json']
                }
            }
        }
        
        dvc_yaml = self.project_root / 'dvc.yaml'
        
        with open(dvc_yaml, 'w') as f:
            yaml.dump(pipeline_config, f, default_flow_style=False)
        
        print(f"✓ Created pipeline configuration: {dvc_yaml}")
        
        # Stage for git
        subprocess.run(['git', 'add', 'dvc.yaml'], cwd=self.project_root)
        print("✓ Pipeline configuration staged for git commit")
    
    def create_params_file(self):
        """Create params.yaml for DVC."""
        print("\nCreating params.yaml...")
        
        params = {
            'prepare_data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'random_seed': 42
            },
            'train_ner': {
                'model_name': 'bert-base-uncased',
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 10,
                'max_length': 128
            },
            'train_classifier': {
                'model_name': 'bert-base-uncased',
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 10,
                'use_focal_loss': True
            }
        }
        
        params_file = self.project_root / 'params.yaml'
        
        with open(params_file, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)
        
        print(f"✓ Created params file: {params_file}")
        
        # Stage for git
        subprocess.run(['git', 'add', 'params.yaml'], cwd=self.project_root)
    
    def setup_metrics(self):
        """Setup metrics tracking."""
        print("\nSetting up metrics tracking...")
        
        # Create metrics directories
        metrics_dirs = ['evaluation', 'models/ner', 'models/classifier']
        
        for metrics_dir in metrics_dirs:
            dir_path = self.project_root / metrics_dir
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✓ Metrics directories created")
    
    def print_usage(self):
        """Print DVC usage instructions."""
        print("\n" + "=" * 60)
        print("DVC Usage Instructions")
        print("=" * 60)
        print("\nCommon DVC commands:")
        print("  dvc status              - Check pipeline status")
        print("  dvc repro               - Reproduce pipeline")
        print("  dvc dag                 - Show pipeline DAG")
        print("  dvc metrics show        - Show metrics")
        print("  dvc metrics diff        - Compare metrics")
        print("  dvc push                - Push data to remote")
        print("  dvc pull                - Pull data from remote")
        print("  dvc run                 - Run a pipeline stage")
        print("\nData versioning:")
        print("  dvc add data/           - Track data directory")
        print("  git add data.dvc        - Commit DVC metadata")
        print("  git commit              - Commit to git")
        print("  dvc push                - Push data to remote")
        print("\nPipeline execution:")
        print("  dvc repro train_ner     - Run specific stage")
        print("  dvc repro               - Run all outdated stages")
        print("=" * 60)
    
    def print_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 60)
        print("DVC Setup Summary")
        print("=" * 60)
        print(f"Project root: {self.project_root}")
        print(f"DVC remotes: {len(self.config.get('remotes', []))}")
        print(f"Data directories: {len(self.config.get('data_dirs', []))}")
        print(f"Cache directory: {self.config.get('cache', {}).get('dir', '.dvc/cache')}")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Commit DVC configuration: git commit -m 'Setup DVC'")
        print("2. Add data: dvc add data/")
        print("3. Run pipeline: dvc repro")
        print("4. Push data: dvc push")
        print("=" * 60)
    
    def run_setup(self, skip_pipeline: bool = False):
        """Run complete DVC setup."""
        print("=" * 60)
        print("DVC Setup")
        print("=" * 60)
        
        try:
            # Check prerequisites
            if not self.check_dvc_installed():
                return False
            
            if not self.check_git_initialized():
                print("\nInitializing git repository...")
                subprocess.run(['git', 'init'], cwd=self.project_root, check=True)
                print("✓ Git initialized")
            
            # Run setup steps
            self.initialize_dvc()
            self.configure_remotes()
            self.setup_data_tracking()
            
            if not skip_pipeline:
                self.create_pipeline()
                self.create_params_file()
            
            self.setup_metrics()
            self.print_usage()
            self.print_summary()
            
            print("\n✓ DVC setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n✗ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup DVC for data versioning and pipeline management"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mlops_config.yaml',
        help='Path to MLOps configuration file'
    )
    parser.add_argument(
        '--skip-pipeline',
        action='store_true',
        help='Skip pipeline creation'
    )
    
    args = parser.parse_args()
    
    setup = DVCSetup(config_path=args.config)
    success = setup.run_setup(skip_pipeline=args.skip_pipeline)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
