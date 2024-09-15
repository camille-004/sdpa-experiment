from sdpa.experiments.api import API


def main() -> None:
    config_path = "sdpa.yaml"
    api = API(config_path)
    api.run()


if __name__ == "__main__":
    main()
