import yaml

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def load_label_map(label_map):
    with open(label_map, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def convert_label_map_name_to_id(label_map):
    if not isinstance(label_map, dict):
        raise ValueError("label map is not dictionary type")
    return dict((_name,_id) for _id, _name in label_map.items())

    