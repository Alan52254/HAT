import csv
import os
import random
import sys

sys.path.insert(0, os.getcwd())

from latency_predictor import LatencyPredictor


OUT_DIR = "../reports/final_eval/search/hat_pipeline_20260609_01/candidates/predictor_only_topk"
CKPT_PATH = "../reports/final_eval/search/hat_pipeline_20260609_01/predictor/latency_predictor_fallback_titanxp.pt"
SUMMARY_PATH = os.path.join(OUT_DIR, "top_candidates_summary.csv")
LATENCY_CONSTRAINT = 90.0
SAMPLE_COUNT = 500


def sample_config():
    encoder_layers = 6
    decoder_layers = 6
    decoder_layer_num = random.choice([6, 5, 4, 3, 2, 1])
    return {
        "encoder": {
            "encoder_embed_dim": random.choice([640, 512]),
            "encoder_layer_num": encoder_layers,
            "encoder_ffn_embed_dim": [random.choice([2048, 1024, 512]) for _ in range(encoder_layers)],
            "encoder_self_attention_heads": [random.choice([4, 2]) for _ in range(encoder_layers)],
        },
        "decoder": {
            "decoder_embed_dim": random.choice([640, 512]),
            "decoder_layer_num": decoder_layer_num,
            "decoder_ffn_embed_dim": [random.choice([2048, 1024, 512]) for _ in range(decoder_layers)],
            "decoder_self_attention_heads": [random.choice([4, 2]) for _ in range(decoder_layers)],
            "decoder_ende_attention_heads": [random.choice([4, 2]) for _ in range(decoder_layers)],
            "decoder_arbitrary_ende_attn": [random.choice([-1, 1, 2]) for _ in range(decoder_layers)],
        },
    }


def write_config(config, path):
    encoder_layer_num = config["encoder"]["encoder_layer_num"]
    decoder_layer_num = config["decoder"]["decoder_layer_num"]
    with open(path, "w") as fid:
        fid.write(f"encoder-embed-dim-subtransformer: {config['encoder']['encoder_embed_dim']}\n")
        fid.write(f"decoder-embed-dim-subtransformer: {config['decoder']['decoder_embed_dim']}\n\n")
        fid.write(f"encoder-ffn-embed-dim-all-subtransformer: {config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num]}\n")
        fid.write(f"decoder-ffn-embed-dim-all-subtransformer: {config['decoder']['decoder_ffn_embed_dim'][:decoder_layer_num]}\n\n")
        fid.write(f"encoder-layer-num-subtransformer: {config['encoder']['encoder_layer_num']}\n")
        fid.write(f"decoder-layer-num-subtransformer: {config['decoder']['decoder_layer_num']}\n\n")
        fid.write(f"encoder-self-attention-heads-all-subtransformer: {config['encoder']['encoder_self_attention_heads'][:encoder_layer_num]}\n")
        fid.write(f"decoder-self-attention-heads-all-subtransformer: {config['decoder']['decoder_self_attention_heads'][:decoder_layer_num]}\n")
        fid.write(f"decoder-ende-attention-heads-all-subtransformer: {config['decoder']['decoder_ende_attention_heads'][:decoder_layer_num]}\n\n")
        fid.write(f"decoder-arbitrary-ende-attn-all-subtransformer: {config['decoder']['decoder_arbitrary_ende_attn'][:decoder_layer_num]}\n\n")


def main():
    random.seed(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    predictor = LatencyPredictor(
        feature_norm=[640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2],
        lat_norm=200,
        ckpt_path=CKPT_PATH,
    )
    predictor.load_ckpt()

    candidates = []
    seen = set()
    for _ in range(SAMPLE_COUNT):
        config = sample_config()
        key = repr(config)
        if key in seen:
            continue
        seen.add(key)
        pred = predictor.predict_lat(config)
        if pred <= LATENCY_CONSTRAINT:
            candidates.append((pred, config))

    # Since validation loss is unavailable in this CPU-only fallback, rank by
    # highest predicted latency within the constraint as a rough capacity proxy.
    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[:5]

    with open(SUMMARY_PATH, "w", newline="") as fid:
        writer = csv.writer(fid)
        writer.writerow(["rank", "predicted_latency_ms", "latency_constraint_ms", "ranking_note", "config_path"])
        for rank, (pred, config) in enumerate(top, start=1):
            config_path = os.path.join(OUT_DIR, f"top_{rank}.yml")
            write_config(config, config_path)
            writer.writerow([rank, pred, LATENCY_CONSTRAINT, "predictor_only_capacity_proxy", config_path])

    print(f"sample_count={SAMPLE_COUNT}")
    print(f"valid_under_constraint={len(candidates)}")
    print(f"summary_path={SUMMARY_PATH}")
    for rank, (pred, config) in enumerate(top, start=1):
        print(f"rank={rank} predicted_latency_ms={pred} config={config}")


if __name__ == "__main__":
    main()
