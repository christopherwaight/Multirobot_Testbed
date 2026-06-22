"""
Load YAML config files for time-varying field parameters.

Each field function in src/fields/environments/ that supports time-varying
behavior exposes a `config_name` attribute. AnalyticalField looks up
config/fields/<config_name>.yaml at construction time and passes the loaded
dict to the field function as the `config` kwarg on each call.

Missing config files return an empty dict, so existing experiments keep
working with the field function's hard-coded defaults.
"""
import os
import yaml


_FIELDS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_FIELDS_DIR, "..", ".."))
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config", "fields")


def load_field_config(config_name):
    """
    Load config/fields/<config_name>.yaml. Returns dict.

    Args:
        config_name: Bare filename (without .yaml extension), e.g. "vortex".

    Returns:
        Dict of parameter values, or empty dict if file does not exist
        or the YAML body is empty.
    """
    if not config_name:
        return {}
    path = os.path.join(_CONFIG_DIR, f"{config_name}.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}
