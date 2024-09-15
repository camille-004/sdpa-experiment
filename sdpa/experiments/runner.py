from sdpa.attention.factory import AttentionFactory
from sdpa.attention.manager import AttentionManager
from sdpa.experiments.config import ExperimentConfig
from sdpa.utils.data_generation import generate_attention_data
from sdpa.utils.types import AttentionConfig


def run_experiment(config: ExperimentConfig):
    results = []
    attention_config: AttentionConfig = {}

    if config.num_heads is not None:
        if config.d_model is None:
            raise ValueError(
                "d_model must be specified when num_heads is provided."
            )
        attention_config["d_model"] = config.d_model
        attention_config["num_heads"] = config.num_heads
    else:
        if config.d_k is None:
            raise ValueError(
                "d_k must be specified when num_heads is not provided."
            )
        attention_config["d_k"] = config.d_k

    attention = AttentionFactory.create(attention_config)

    manager = AttentionManager()
    manager.register_attention("experiment_attention", attention)

    for scale in config.scaling_factors:
        if config.num_heads is not None:
            queries, keys, values = generate_attention_data(
                batch_size=config.batch_size,
                seq_length=config.seq_length,
                d_model=config.d_model,
            )
        else:
            queries, keys, values = generate_attention_data(
                batch_size=config.batch_size,
                seq_length=config.seq_length,
                d_k=config.d_k,
            )

        attention_menchanism = manager.get_attention("experiment_attention")

        scaled_queries = queries * scale

        output = attention_menchanism.forward(scaled_queries, keys, values)
        results.append({"scale": scale, "weights": output.weights.tolist()})

    return results
