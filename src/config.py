import yaml
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carga y retorna el archivo de configuracion"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)