import logging
from typing import Dict, List, Optional, Tuple

try:
    import onnxruntime as ort
except ImportError:
    pass  # Let the main entry point handle the fatal error

from config import Config

logger = logging.getLogger("FaceSystem.ONNX")

def build_session_options(cfg: Config) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = cfg.ort_intra_threads
    opts.inter_op_num_threads = cfg.ort_inter_threads
    opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts

def build_providers(cfg: Config) -> Tuple[List, Optional[Dict]]:
    if not cfg.use_gpu:
        return ["CPUExecutionProvider"], None
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" not in avail:
        logger.warning("CUDA unavailable — falling back to CPU.")
        return ["CPUExecutionProvider"], None
    cuda_opts = {
        "device_id":              "0",
        "arena_extend_strategy":  "kSameAsRequested",
        "gpu_mem_limit":          str(4 * 1024 ** 3),
        "cudnn_conv_algo_search": "HEURISTIC",
    }
    logger.info("Using CUDAExecutionProvider")
    return ["CUDAExecutionProvider", "CPUExecutionProvider"], cuda_opts

def make_session(model_path: str, cfg: Config,
                 providers: List, cuda_opts: Optional[Dict]) -> ort.InferenceSession:
    opts = build_session_options(cfg)
    if cuda_opts and "CUDAExecutionProvider" in providers:
        provider_options = [cuda_opts if p == "CUDAExecutionProvider" else {}
                            for p in providers]
        return ort.InferenceSession(model_path, sess_options=opts,
                                    providers=providers,
                                    provider_options=provider_options)
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)

def validate_model_shapes(session: ort.InferenceSession, label: str,
                           expected_inputs: List[Tuple], min_outputs: int) -> None:
    """Shape check at startup. Skips dynamic dims (-1, None, str)."""
    outputs = session.get_outputs()
    if len(outputs) < min_outputs:
        raise ValueError(f"{label}: expected ≥{min_outputs} outputs, "
                         f"got {len(outputs)}: {[o.name for o in outputs]}")
    for (name_substr, exp_shape), inp in zip(expected_inputs, session.get_inputs()):
        if name_substr and name_substr not in inp.name:
            raise ValueError(f"{label}: input '{inp.name}' missing '{name_substr}'")
        if exp_shape:
            for i, (ed, ad) in enumerate(zip(exp_shape, inp.shape)):
                if ed == -1 or ad is None or isinstance(ad, str):
                    continue
                if ed != ad:
                    raise ValueError(f"{label}: dim[{i}] expected {ed}, got {ad}")
    logger.info("%s: shape OK (%d outputs).", label, len(outputs))
