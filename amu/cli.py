import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Road-scene assignment-matched utility supervision")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train-probe", "labels", "train-router", "train-detection-router", "evaluate", "predict", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cpu", help="cpu, cuda:0 or mps; use 0 for the Ultralytics probe trainer")
        if name != "train-probe":
            p.add_argument("--checkpoint", required=True)
        if name != "benchmark":
            p.add_argument("--output", required=True, help="New directory; existing results are never overwritten")
        if name in ("train-probe", "labels", "train-detection-router", "evaluate"):
            p.add_argument("--data", required=True)
        if name != "train-router":
            p.add_argument("--imgsz", type=int, default=640)
        if name in ("train-probe", "labels", "train-router", "train-detection-router"):
            p.add_argument("--seed", type=int, default=17)
        if name in ("train-probe", "train-router", "train-detection-router"):
            p.add_argument("--epochs", type=int, default=300 if name == "train-probe" else 30)
            p.add_argument("--batch", type=int, default=32)
        if name == "train-probe":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--block", type=int, default=8)
        elif name == "labels":
            p.add_argument("--draws", type=int, default=2)
            p.add_argument("--regions", type=int, default=4)
            p.add_argument("--shard-size", type=int, default=4096)
            p.add_argument("--limit", type=int, help="Pipeline smoke test only; not a full paper run")
        elif name == "train-router":
            p.add_argument("--labels", required=True)
            p.add_argument("--target", choices=["symmetric", "reassigned", "on", "off"], default="symmetric")
            p.add_argument("--shuffled", action="store_true")
        elif name == "evaluate":
            p.add_argument("--batch", type=int, default=1)
            p.add_argument("--mode", choices=["router", "on", "off"])
        elif name == "predict":
            p.add_argument("--source", required=True)
            p.add_argument("--conf", type=float, default=.25)
        elif name == "benchmark":
            p.add_argument("--warmup", type=int, default=200)
            p.add_argument("--iterations", type=int, default=1000)
            p.add_argument("--rounds", type=int, default=5)
    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    for key in ("epochs", "batch", "imgsz", "block", "draws", "regions", "shard_size", "iterations", "rounds", "limit"):
        if key in kwargs and kwargs[key] is not None and kwargs[key] <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    from .training import train_probe, generate_labels, train_router, train_detection_router
    from .evaluation import evaluate, predict, benchmark
    functions = {"train-probe": train_probe, "labels": generate_labels, "train-router": train_router,
                 "train-detection-router": train_detection_router, "evaluate": evaluate,
                 "predict": predict, "benchmark": benchmark}
    if command in ("train-router", "train-detection-router"):
        kwargs["batch_size"] = kwargs.pop("batch")
    result = functions[command](**kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
