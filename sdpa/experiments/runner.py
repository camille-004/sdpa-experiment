from sdpa.attention.factory import AttentionFactory
from sdpa.attention.manager import AttentionManager
from sdpa.experiments.config import ExperimentConfig
from sdpa.utils.data_generation import generate_attention_data
from sdpa.utils.types import AttentionConfig, AttentionType


def run_experiment(config: ExperimentConfig):
    results = []
    attention_config: AttentionConfig = {
        "d_model": config.d_model,
        "num_heads": config.num_heads,
    }

    attention = AttentionFactory.create(
        AttentionType.SCALED_DOT_PRODUCT, attention_config
    )

    manager = AttentionManager()
    manager.register_attention("experiment_attention", attention)

    for scale in config.scaling_factors:
        queries, keys, values = generate_attention_data(
            batch_size=config.batch_size,
            seq_length=config.seq_length,
            d_k=config.d_k,
            d_v=config.d_v,
        )

        attention_menchanism = manager.get_attention("experiment_attention")

        scaled_queries = queries * scale

        output = attention_menchanism.forward(
            scaled_queries[0], keys[0], values[0]
        )

        results.append({"scale": scale, "weights": output.weights})

    return results
