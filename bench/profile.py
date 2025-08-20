"""
Profiling utilities: params, FLOPs, latency for the model.
"""
import torch
import time
from models.build import build_model

def count_params(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())

def profile_latency(model, input_size=(2,3,256,256), iters=30):
    """Profile average latency (ms) over N runs."""
    model.eval()
    x = torch.randn(*input_size)
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(iters):
            _ = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
    return 1000 * (end - start) / iters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--img', nargs=2, type=int, default=[256,256])
    args = parser.parse_args()
    model = build_model(args.config)
    params = count_params(model)
    print(f"Params: {params/1e6:.2f}M")
    latency = profile_latency(model, input_size=(2,3,args.img[0],args.img[1]))
    print(f"Latency: {latency:.2f} ms (batch=2)")
    # FLOPs: requires ptflops or fvcore, not included by default
    try:
        from ptflops import get_model_complexity_info
        macs, params2 = get_model_complexity_info(model, (3, args.img[0], args.img[1]), as_strings=True, print_per_layer_stat=False)
        print(f"FLOPs: {macs}")
    except ImportError:
        print("Install ptflops for FLOPs counting.")
