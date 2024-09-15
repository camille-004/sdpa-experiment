#!/bin/bash

CONFIG_DIR="configs"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory '$CONFIG_DIR' not found."
    exit 1
fi

CONFIG_FILES=$(find "$CONFIG_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \))

if [ -z "$CONFIG_FILES" ]; then
    echo "Error: No YAML config files found in '$CONFIG_DIR' or its subdirectories."
    exit 1
fi

for CONFIG_FILE in $CONFIG_FILES; do
    echo "Running experiment with config: $CONFIG_FILE"
    python main.py "$CONFIG_FILE"

    # shellcheck disable=SC2181
    if [ $? -eq 0 ]; then
        echo "Experiment completed successfully."
    else
        echo "Error: Experiment failed for config: $CONFIG_FILE"
    fi

    echo "----------------------------------------"
done

echo "All experiments completed."
