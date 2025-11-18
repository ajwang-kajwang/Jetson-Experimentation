#!/usr/bin/env -S python3 -u
import tensorrt as trt, argparse
from pathlib import Path

def parse_shape(s): return tuple(map(int, s.lower().split("x")))

def main():
    ap = argparse.ArgumentParser(description="ONNX → TensorRT engine converter (minimal)")
    ap.add_argument("onnx", type=Path, help=".onnx file")
    ap.add_argument("-o", "--output", type=Path, help="output .engine path")
    ap.add_argument("--fp32", action="store_true", help="use FP32 (default FP16)")
    ap.add_argument("--input", type=str, help="input tensor name")
    ap.add_argument("--min", type=parse_shape, help="min shape e.g. 1x3x640x640")
    ap.add_argument("--opt", type=parse_shape, help="opt shape e.g. 1x3x640x640")
    ap.add_argument("--max", type=parse_shape, help="max shape e.g. 4x3x640x640")
    ap.add_argument("--workspace", type=int, default=4, help="workspace size GB")
    args = ap.parse_args()

    if not args.onnx.exists(): raise SystemExit("ONNX file not found")
    eng = args.output or args.onnx.with_suffix((".fp32" if args.fp32 else ".fp16") + ".engine")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(args.onnx.read_bytes()):
        [print(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("Failed to parse ONNX")

    cfg = builder.create_builder_config()
    ws = args.workspace * (1 << 30)
    try: cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    except AttributeError: cfg.max_workspace_size = ws
    if not args.fp32 and builder.platform_has_fast_fp16: cfg.set_flag(trt.BuilderFlag.FP16)

    t = network.get_input(0)
    if any(d == -1 for d in t.shape) or any([args.min, args.opt, args.max]):
        prof = builder.create_optimization_profile()
        n = args.input or t.name
        min_s = args.min or (1,3,640,640); opt_s = args.opt or min_s; max_s = args.max or (4,3,640,640)
        prof.set_shape(n, min_s, opt_s, max_s)
        cfg.add_optimization_profile(prof)

    engine = builder.build_engine(network, cfg)
    if not engine: raise SystemExit("Engine build failed")
    eng.write_bytes(engine.serialize())
    print(f"✅ Saved {eng} ({eng.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__": main()
