import argparse

from sdpa.experiments.experiment import Experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPDA experiment")
    parser.add_argument(
        "config_path", type=str, help="Path to the config YAML file"
    )
    args = parser.parse_args()

    experiment = Experiment(args.config_path)
    experiment.run()


if __name__ == "__main__":
    main()
