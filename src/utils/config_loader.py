import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Configuration Loader: Support YAML configuration files and inheritance mechanism"""

    def __init__(self, config_path: str, _visited=None):
        """
        Initialize Configuration Loader

        Args:
            config_path: Configuration filepath
            _visited: Internal useforDetectCircular dependency
        """
        if _visited is None:
            _visited = set()

        self.config_path = Path(config_path).resolve()

        # DetectCircular dependency
        if str(self.config_path) in _visited:
            raise ValueError(
                f"DetecttoCircular dependency: {self.config_path}\n"
                f"Already visited configuration file: {' -> '.join(_visited)}"
            )

        _visited.add(str(self.config_path))
        self.config = self._Load_config(_visited)

    def _Load_config(self, _visited: set) -> Dict[str, Any]:
        """
        LoadConfiguration fileSupportbase_configInheritance

        Args:
            _visited: Set of already visited configuration files, for detecting circular dependencies
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_Load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file does not exist: {self.config_path}\n"
                f"Please check if path is correct"
            )

        # If base_config exists, load base configuration first
        if 'experiment' in config and 'base_config' in config['experiment']:
            base_path = self.config_path.parent / config['experiment']['base_config']
            base_config = ConfigLoader(str(base_path), _visited).config

            # Deep mergeconfiguration
            config = self._deep_merge(base_config, config)

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep mergeTwo dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key_path: str, default=None):
        """Get configuration value, support dot separated path (e.g. 'training.episodes')"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def validate_ablation(self) -> bool:
        """Validate reasonableness of ablation experiment configuration"""
        issues = []
        warnings = []

        # check1ResamplingMustdisabled
        if self.get('resampling.enabled', False):
            issues.append(" Resampling not disabled (Ablation experiment must disable it)")

        # check2ensureOnly modifyOne componentSpecialhandleclass weight
        base_components = {
            'use_sparse_vd': True,
            'hard_negative': True,
            'use_fixed_pos_weight': True,
            'threshold_mode': 'search',
            'meta_learning': True
        }

        current_components = self.config.get('components', {})

        # Calculate modified component count (special handle class weight)
        diff_details = []
        for k, v in base_components.items():
            current_val = current_components.get(k)

            # Special handle: class weight ablation check fixed_pos_weight value
            if k == 'use_fixed_pos_weight' and current_val == True:
                fixed_weight = current_components.get('fixed_pos_weight', 3.0)
                if fixed_weight != 3.0:
                    diff_details.append(f"Class Weight (fixed_pos_weight={fixed_weight})")
            elif current_val != v:
                diff_details.append(k)

        diff_count = len(diff_details)

        if diff_count > 1:
            issues.append(f" Modified {diff_count} components (Standard ablation should only modify 1)")
            warnings.extend([f"    - {d}" for d in diff_details])
        elif diff_count == 0:
            warnings.append("ℹ  This isBaselineconfigurationUnmodifiedAnycomponent")
        else:
            warnings.append(f"[OK] Ablationtarget: {diff_details[0]}")

        # printvalidateresult
        config_name = self.get('experiment.name', self.config_path.name)
        print(f"\n{config_name}")

        for w in warnings:
            print(f"  {w}")

        if issues:
            for issue in issues:
                print(f"  {issue}")
            return diff_count <= 1 and not self.get('resampling.enabled', False)
        else:
            print("  [OK] configurationvalidatethrough")
            return True

if __name__ == "__main__":
    # Simple self-test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python config_loader.py <config.yaml>")
    else:
        cfg = ConfigLoader(path)
        print("Load success, example: training.episodes =", cfg.get('training.episodes'))
        cfg.validate_ablation()
