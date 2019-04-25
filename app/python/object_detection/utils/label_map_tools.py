import yaml

def load_label_map(label_map_file):
    with open(label_map_file, 'r') as stream:
        try:
            label_map = yaml.safe_load(stream)
            attribute = list(label_map.keys())[0]
            label_id_to_name = label_map[attribute]
            return attribute, label_id_to_name
        except yaml.YAMLError as exc:
            print(exc)

def convert_label_map(label_id_to_name):
    if not isinstance(label_id_to_name, dict):
        raise ValueError("label map is not dictionary type")
    return dict((_name,_id) for _id, _name in label_id_to_name.items())
