import yaml


def load_config(path):
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(f"Loaded config file from {path}", end="\n\n")
            for key, value in config.items():
                print(f"{key}: {value}")
            return config
        except yaml.YAMLError as exc:
            print(exc)
