import yaml
import os

def load_config(config_name):
    config_path = os.path.join('src/configs', f'{config_name}.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

database_config = load_config('database_config')