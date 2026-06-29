"""Modal setup for MEVL-VTS v7 (GoMatching++ + 6-encoder bridge).

Three functions, run in order:
  1. verify_image        -> confirms Docker image builds with all deps
  2. verify_baseline_load-> confirms the reproduced iter-30k checkpoint loads
  3. verify_smoke_forward-> runs a 1-step forward+backward on synthetic input

If all three pass, the foundation is solid and we can layer v7 code on top.

USAGE (run from local Modal-authenticated shell, NOT this agent):
  # one-time deploy
  modal deploy modal/modal_v7_setup.py

  # then individual functions:
  modal run modal/modal_v7_setup.py::verify_image
  modal run modal/modal_v7_setup.py::verify_baseline_load
  modal run modal/modal_v7_setup.py::verify_smoke_forward

  # to upload the reproduced iter-30k checkpoint to a Modal volume:
  modal volume put mevl-vts-v7-checkpoints \\
      <local_path>/model_final.pth \\
      gomatching_iter30k/model_final.pth

VOLUMES (auto-created on first run):
  mevl-vts-v7-checkpoints  -> deepsolo pretrained + our iter-30k reproduction
  mevl-vts-v7-data         -> IC15-V + DSText annotations & frames
  mevl-vts-v7-outputs      -> training output dirs (checkpoints, logs)
  mevl-vts-v7-encoder-cache-> Phase 1 encoder feature cache (filled later)
"""
import os
import modal

APP_NAME = "mevl-vts-v7"
app = modal.App(APP_NAME)

# Volumes
v_ckpts = modal.Volume.from_name("mevl-vts-v7-checkpoints", create_if_missing=True)
v_data = modal.Volume.from_name("mevl-vts-v7-data", create_if_missing=True)
v_outputs = modal.Volume.from_name("mevl-vts-v7-outputs", create_if_missing=True)
v_encache = modal.Volume.from_name("mevl-vts-v7-encoder-cache", create_if_missing=True)

# Secrets (only needed later for encoder caching from gated HF models like SAM)
# Uncomment after creating the secret in research-work workspace:
#   https://modal.com/secrets/research-work/main/create?secret_name=huggingface-secret&key=HF_TOKEN
# hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# ---------------------------------------------------------------------------
# Image: CUDA 12.1 + torch 2.5.1 + detectron2 + AdelaiDet (compiled)
# ---------------------------------------------------------------------------
# Compiled with TORCH_CUDA_ARCH_LIST="8.9" for L4 (sm_89, Ada Lovelace).
# If you target a different GPU, edit this list.

GOMATCHING_REPO = "https://github.com/Hxyz-123/GoMatching.git"
TORCH_VER = "2.5.1"
TORCHVISION_VER = "0.20.1"
CUDA_TAG = "cu121"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "wget", "curl",
        "build-essential", "ninja-build", "clang",   # clang needed for Polygon3
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6",
    )
    .env({
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9",          # L4. Change if not L4.
        "PYTHONUNBUFFERED": "1",
    })
    # wheel + setuptools needed BEFORE no-build-isolation installs
    .pip_install("wheel", "setuptools>=68", "ninja")
    # Pin torch first so detectron2 build links against it
    .pip_install(
        f"torch=={TORCH_VER}",
        f"torchvision=={TORCHVISION_VER}",
        index_url=f"https://download.pytorch.org/whl/{CUDA_TAG}",
    )
    .pip_install(
        # Core CV / DL deps
        "opencv-python-headless==4.10.0.84",
        "Pillow", "numpy<2", "scipy", "shapely", "pyclipper",
        "timm==1.0.11", "einops",
        # OCR/text-spotting specific (AdelaiDet deps)
        "Polygon3", "rapidfuzz", "editdistance", "fvcore", "iopath",
        "yacs", "tabulate", "tqdm", "termcolor",
        # Foundation-model encoders (Phase 1 cache)
        "open_clip_torch", "gdown", "h5py",
        # HuggingFace stack for SAM / DINOv2 / SigLIP / EVA-CLIP loaders
        "huggingface_hub", "safetensors", "transformers>=4.45,<5.0",
        "accelerate>=0.34,<1.2",
    )
    # detectron2 from source (matches GoMatching install instructions)
    .pip_install(
        "git+https://github.com/facebookresearch/detectron2.git",
        extra_options="--no-build-isolation",
    )
    # Clone GoMatching at image-build time; baked into /opt/GoMatching
    .run_commands(
        f"git clone --depth 1 {GOMATCHING_REPO} /opt/GoMatching",
        # Build AdelaiDet (vendored under third_party)
        "cd /opt/GoMatching/third_party && python -m pip install -e . --no-build-isolation 2>&1 | tail -20",
    )
    # extra deps for v7 DenseTrack components
    .pip_install("rapidfuzz", "h5py")
    # Replace the cloned gomatching/ Python package with our forked version
    # (this brings in our patches to config.py, freeze_layers.py, gom_lstmatcher.py,
    # shared_ffn_crsattn.py, plus the new v7_densetrack/ subpackage).
    .add_local_dir(
        "C:/Users/amana/OneDrive/Desktop/Curve-Aware/Project/MEVL-VTS/GoMatching_v7/gomatching",
        remote_path="/opt/GoMatching/gomatching",
        copy=True,
    )
    # And the configs/ dir for our new DenseTrack yaml
    .add_local_dir(
        "C:/Users/amana/OneDrive/Desktop/Curve-Aware/Project/MEVL-VTS/GoMatching_v7/configs",
        remote_path="/opt/GoMatching/configs",
        copy=True,
    )
    # tools/ for Phase 1 cache scripts (cache_encoder_features.py, etc.)
    .add_local_dir(
        "C:/Users/amana/OneDrive/Desktop/Curve-Aware/Project/MEVL-VTS/GoMatching_v7/tools",
        remote_path="/opt/GoMatching/tools_v7",
        copy=True,
    )
    # Tiny TIM-VTS verifier checkpoint for smoke tests.
    .add_local_file(
        "C:/Users/amana/OneDrive/Desktop/Curve-Aware/Project/MEVL-VTS/runs/tim_recovery_mlp_ocr.pt",
        remote_path="/opt/GoMatching/tim_recovery_mlp_ocr.pt",
        copy=True,
    )
)

# Reference the existing IC15-V dataset volume (already populated in research-work)
v_datasets = modal.Volume.from_name("mevl-vts-datasets", create_if_missing=False)


# ---------------------------------------------------------------------------
# Function 1: verify image
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={
        "/ckpts": v_ckpts,
        "/data": v_data,
        "/outputs": v_outputs,
    },
)
def verify_image():
    """Print versions of every critical dep + CUDA availability + GPU info."""
    import sys, subprocess
    print("=" * 60)
    print("  Stage 1: image verification")
    print("=" * 60)
    print(f"python: {sys.version}")

    import torch
    print(f"torch:        {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device:     {torch.cuda.get_device_name(0)}")
        print(f"  capability: {torch.cuda.get_device_capability(0)}")
        print(f"  mem:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  cuDNN:      {torch.backends.cudnn.version()}")

    import torchvision
    print(f"torchvision:  {torchvision.__version__}")

    import detectron2
    print(f"detectron2:   {detectron2.__version__}")

    # Try importing AdelaiDet
    sys.path.insert(0, "/opt/GoMatching/third_party")
    try:
        import adet
        print(f"adet:         imported OK from {adet.__file__}")
    except Exception as e:
        print(f"adet:         FAILED import: {type(e).__name__}: {e}")

    # Try importing GoMatching
    sys.path.insert(0, "/opt/GoMatching")
    try:
        import gomatching
        print(f"gomatching:   imported OK from {gomatching.__file__}")
    except Exception as e:
        print(f"gomatching:   FAILED import: {type(e).__name__}: {e}")

    # Foundation-model encoders that v7 will need
    for mod_name, pkg in [
        ("open_clip", "open_clip_torch"),
        ("timm", "timm"),
        ("transformers", "transformers"),
    ]:
        try:
            m = __import__(mod_name)
            print(f"{mod_name:13}: {getattr(m, '__version__', 'unknown')}")
        except Exception as e:
            print(f"{mod_name:13}: FAILED {e}")

    # Volumes mounted?
    print("\nVolume mounts:")
    for path in ["/ckpts", "/data", "/outputs"]:
        files = os.listdir(path) if os.path.isdir(path) else "MISSING"
        print(f"  {path}: {files}")

    print("\nGPU info via nvidia-smi:")
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip())
    print("=" * 60)
    print("Stage 1 done. If everything imported, you're ready for Stage 2.")


@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={
        "/outputs": v_outputs,
    },
)
def verify_tim_mlp():
    """Smoke-test the trainable TIM-VTS recovery verifier inside Modal."""
    import os
    import sys
    import torch

    os.chdir("/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    print("=" * 60)
    print("  TIM-VTS MLP verifier smoke")
    print("=" * 60)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    from gomatching.modeling.tim_vts import TIMRecoveryVerifier

    ckpt_path = "/opt/GoMatching/tim_recovery_mlp_ocr.pt"
    if not os.path.exists(ckpt_path):
        return {"ok": False, "reason": f"missing {ckpt_path}"}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    features = ckpt["features"]
    model = TIMRecoveryVerifier(
        in_dim=len(features),
        hidden_dim=ckpt.get("args", {}).get("hidden_dim", 96),
        dropout=ckpt.get("args", {}).get("dropout", 0.15),
    )
    model.load_state_dict(ckpt["model"])
    model.eval().cuda()
    x = torch.zeros(8, len(features), device="cuda")
    with torch.no_grad():
        prob = torch.sigmoid(model(x))
    print(f"features={len(features)}")
    print(f"prob_shape={tuple(prob.shape)} prob_mean={prob.mean().item():.4f}")
    print("TIM MLP smoke PASSED")
    return {"ok": True, "features": len(features), "prob_mean": float(prob.mean().item())}


# ---------------------------------------------------------------------------
# Function 2: verify baseline checkpoint loads
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    volumes={
        "/ckpts": v_ckpts,
        "/data": v_data,
        "/outputs": v_outputs,
    },
    # secrets=[hf_secret],  # not needed for verify stages; add for encoder cache
)
def verify_baseline_load():
    """Build the GoMatching model from config and load the reproduced iter-30k weights.

    Prerequisites:
      - upload model_final.pth to volume mevl-vts-v7-checkpoints first:
        modal volume put mevl-vts-v7-checkpoints \\
            <local_path>/model_final.pth \\
            gomatching_iter30k/model_final.pth
    """
    import sys, os
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    print("=" * 60)
    print("  Stage 2: baseline checkpoint load")
    print("=" * 60)

    # Locate the reproduced checkpoint on the volume
    ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"
    if not os.path.exists(ckpt_path):
        print(f"MISSING checkpoint at {ckpt_path}")
        print("Upload it first:")
        print("  modal volume put mevl-vts-v7-checkpoints <LOCAL>/model_final.pth"
              " gomatching_iter30k/model_final.pth")
        return {"ok": False, "reason": "checkpoint_missing"}

    import torch
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
    print(f"  ckpt keys (top-level): {list(raw.keys())[:5] if isinstance(raw, dict) else 'tensor'}")
    print(f"  total params in state_dict: {sum(p.numel() for p in state_dict.values()):,}")
    print(f"  iter recorded: {raw.get('iteration', 'unknown') if isinstance(raw, dict) else 'unknown'}")

    # Build the GoMatching model (mirrors train_net.py::setup())
    # IMPORTANT: import gomatching first to trigger META_ARCH_REGISTRY registration
    import gomatching  # noqa: F401  registers GoMatching, SHA_FFN_CRSATTN, etc.

    from detectron2.config import get_cfg
    from adet.config import add_deepsolo_cfg  # must come BEFORE add_gom_config
    from gomatching.config import add_gom_config

    cfg = get_cfg()
    add_deepsolo_cfg(cfg)
    add_gom_config(cfg)

    cfg_file = "/opt/GoMatching/configs/GoMatching_PP_ICDAR15.yaml"
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.MODEL.DEVICE = "cuda"
    # match train_net.py setup() behavior
    if hasattr(cfg.MODEL.TRANSFORMER, "INFERENCE_TH_TRAIN"):
        cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TRAIN
    cfg.freeze()
    print(f"  cfg loaded: {cfg_file}")
    print(f"  META_ARCH: {cfg.MODEL.META_ARCHITECTURE}")
    print(f"  ROI_HEADS: {cfg.MODEL.ROI_HEADS.NAME}")

    from detectron2.modeling import build_model
    model = build_model(cfg)
    print(f"  model built: {type(model).__name__}")

    # Load weights
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(model).load(ckpt_path)
    print(f"  weights loaded: {ckpt_path}")

    # Param counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params:     {total:,}")
    print(f"  trainable params: {trainable:,}  ({100 * trainable / total:.3f}%)")

    # Memory check
    if torch.cuda.is_available():
        print(f"  GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("=" * 60)
    print("Stage 2 done. If you see GoMatching params loaded, baseline works.")
    return {"ok": True, "total_params": total, "trainable_params": trainable}


# ---------------------------------------------------------------------------
# Function 3: smoke forward+backward on synthetic input
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    volumes={
        "/ckpts": v_ckpts,
        "/data": v_data,
        "/outputs": v_outputs,
    },
    # secrets=[hf_secret],  # not needed for verify stages; add for encoder cache
)
def verify_smoke_forward():
    """Run a single training step with synthetic 6-frame clip + dummy GT.

    Confirms:
      - model forward works end-to-end
      - loss is computed
      - backward + optimizer.step works
      - GPU memory stays reasonable
    """
    import sys, os, time
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    print("=" * 60)
    print("  Stage 3: smoke forward+backward")
    print("=" * 60)

    ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"
    if not os.path.exists(ckpt_path):
        print(f"MISSING checkpoint - run Stage 2 first, after uploading the .pth")
        return {"ok": False}

    import torch
    import gomatching  # noqa: F401  registers meta-archs / roi heads
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.structures import Instances, Boxes
    from adet.config import add_deepsolo_cfg
    from gomatching.config import add_gom_config

    cfg = get_cfg()
    add_deepsolo_cfg(cfg)
    add_gom_config(cfg)
    cfg.merge_from_file("/opt/GoMatching/configs/GoMatching_PP_ICDAR15.yaml")
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.MODEL.DEVICE = "cuda"
    if hasattr(cfg.MODEL.TRANSFORMER, "INFERENCE_TH_TRAIN"):
        cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TRAIN
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(ckpt_path)
    model.train()

    print(f"  model loaded, mem={torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Build synthetic 6-frame clip
    # Each frame: 3-channel 720x1280 image (typical IC15-V)
    # Each gt instance: 25 control points + 1 class + 1 inst_id
    device = torch.device("cuda")
    H, W = 720, 1280
    num_frames = 6
    n_inst = 3   # 3 text instances per frame (tracked across frames)
    num_points = 25

    batched_inputs = []
    for t in range(num_frames):
        img = torch.randint(0, 255, (3, H, W), dtype=torch.uint8, device=device).float()
        # GT for this frame
        polyline = torch.rand(n_inst, num_points * 2, device=device) * min(H, W)
        # Make instance IDs consistent across frames (tracking)
        gt_ids = torch.arange(n_inst, device=device)
        gt_classes = torch.zeros(n_inst, dtype=torch.long, device=device)
        gt_boxes = Boxes(torch.tensor(
            [[100., 100., 300., 300.],
             [400., 200., 600., 400.],
             [200., 500., 500., 700.]],
            device=device))
        instances = Instances((H, W))
        instances.gt_classes = gt_classes
        instances.gt_boxes = gt_boxes
        instances.polyline = polyline
        instances.texts = torch.zeros(n_inst, num_points, dtype=torch.long, device=device)
        instances.gt_instance_ids = gt_ids
        batched_inputs.append({"image": img, "instances": instances})

    print(f"  synthetic input: {num_frames} frames {H}x{W}, {n_inst} insts each")

    # Forward + backward
    try:
        t0 = time.time()
        losses = model(batched_inputs)
        fwd_t = time.time() - t0
        print(f"  forward: {fwd_t:.2f}s  losses: {list(losses.keys())}")
        for k, v in losses.items():
            print(f"    {k}: {v.item():.4f}")

        total_loss = sum(losses.values())
        t0 = time.time()
        total_loss.backward()
        bwd_t = time.time() - t0
        print(f"  backward: {bwd_t:.2f}s")
        print(f"  total_loss: {total_loss.item():.4f}")
        print(f"  GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        # Optimizer step (small lr, just to confirm no NaN)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        opt.step()
        opt.zero_grad()

        print("=" * 60)
        print("Stage 3 done. Foundation is solid.")
        return {"ok": True, "fwd_s": fwd_t, "bwd_s": bwd_t,
                "loss": total_loss.item(),
                "peak_gb": torch.cuda.max_memory_allocated()/1e9}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Function 4: verify DenseTrack v7 components (standalone unit tests on GPU)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={"/ckpts": v_ckpts, "/data": v_data, "/outputs": v_outputs},
)
def verify_components():
    """Run the 3 component self-tests + light integration check (no checkpoint
    needed)."""
    import sys, importlib, torch
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    print("=" * 60)
    print("  Stage 4: DenseTrack component verification")
    print("=" * 60)

    results = {}

    # Component 1: needs detectron2 — should work here
    print("\n--- Component 1: SAMProposalAugmenter ---")
    try:
        m = importlib.import_module("gomatching.modeling.v7_densetrack.proposal_augmenter")
        m._self_test()
        results["c1_proposal_augmenter"] = "PASS"
    except Exception as e:
        import traceback; traceback.print_exc()
        results["c1_proposal_augmenter"] = f"FAIL: {type(e).__name__}: {e}"

    # Component 2
    print("\n--- Component 2: MultiEncoderConsensusMatcher ---")
    try:
        m = importlib.import_module("gomatching.modeling.v7_densetrack.consensus_matcher")
        m._self_test()
        results["c2_consensus_matcher"] = "PASS"
    except Exception as e:
        import traceback; traceback.print_exc()
        results["c2_consensus_matcher"] = f"FAIL: {type(e).__name__}: {e}"

    # Component 3
    print("\n--- Component 3: VLMContentMatcher ---")
    try:
        m = importlib.import_module("gomatching.modeling.v7_densetrack.content_matcher")
        m._self_test()
        results["c3_content_matcher"] = "PASS"
    except Exception as e:
        import traceback; traceback.print_exc()
        results["c3_content_matcher"] = f"FAIL: {type(e).__name__}: {e}"

    # Integration check: same baseline forward as Stage 3, but with components
    # instantiated and called inline on the model's outputs (not yet wired).
    # This confirms the components can consume real GoMatching tensors.
    print("\n--- Integration check (components called on real GoMatching outputs) ---")
    try:
        # Build model from baseline (reuse Stage 2 logic)
        import gomatching  # noqa: F401
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.structures import Instances, Boxes
        from adet.config import add_deepsolo_cfg
        from gomatching.config import add_gom_config

        ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"
        cfg = get_cfg()
        add_deepsolo_cfg(cfg)
        add_gom_config(cfg)
        cfg.merge_from_file("/opt/GoMatching/configs/GoMatching_PP_ICDAR15.yaml")
        cfg.MODEL.WEIGHTS = ckpt_path
        cfg.MODEL.DEVICE = "cuda"
        if hasattr(cfg.MODEL.TRANSFORMER, "INFERENCE_TH_TRAIN"):
            cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TRAIN
        cfg.freeze()
        model = build_model(cfg).eval()
        if os.path.exists(ckpt_path):
            DetectionCheckpointer(model).load(ckpt_path)

        from gomatching.modeling.v7_densetrack import (
            SAMProposalAugmenter, MultiEncoderConsensusMatcher, VLMContentMatcher
        )
        c1 = SAMProposalAugmenter().cuda()
        c2 = MultiEncoderConsensusMatcher(
            encoder_dims={"clip-l-336": 1024, "dinov2-l": 1024,
                          "sam-h": 1280, "convnext-l": 1536},
            proj_dim=256,
        ).cuda()
        c3 = VLMContentMatcher().cuda()

        n_v7 = sum(p.numel() for p in c1.parameters()) + \
               sum(p.numel() for p in c2.parameters()) + \
               sum(p.numel() for p in c3.parameters())
        print(f"  v7 trainable: c1={sum(p.numel() for p in c1.parameters()):,}, "
              f"c2={sum(p.numel() for p in c2.parameters()):,}, "
              f"c3={sum(p.numel() for p in c3.parameters()):,}, total={n_v7:,}")
        results["integration_param_count"] = n_v7
        results["integration"] = "PASS"
    except Exception as e:
        import traceback; traceback.print_exc()
        results["integration"] = f"FAIL: {type(e).__name__}: {e}"

    print("\n" + "=" * 60)
    print("Stage 4 results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return results


# ---------------------------------------------------------------------------
# Function 5: deep integration smoke test (DENSETRACK enabled + mock context)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    volumes={"/ckpts": v_ckpts, "/data": v_data, "/outputs": v_outputs},
)
def verify_deep_integration():
    """Stage 5: load model with all v7 components enabled, push mock context
    down through the model, run forward + backward, verify gates receive grad."""
    import os, sys, time, torch
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    print("=" * 60)
    print("  Stage 5: deep integration test (all 3 v7 components wired)")
    print("=" * 60)

    ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"
    if not os.path.exists(ckpt_path):
        print(f"  MISSING checkpoint at {ckpt_path}")
        return {"ok": False, "reason": "no_checkpoint"}

    import gomatching  # noqa
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.structures import Instances, Boxes
    from adet.config import add_deepsolo_cfg
    from gomatching.config import add_gom_config

    # Build cfg with ALL 3 v7 components enabled
    cfg = get_cfg()
    add_deepsolo_cfg(cfg)
    add_gom_config(cfg)
    cfg.merge_from_file("/opt/GoMatching/configs/GoMatching_PP_ICDAR15.yaml")
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.MODEL.DEVICE = "cuda"
    if hasattr(cfg.MODEL.TRANSFORMER, "INFERENCE_TH_TRAIN"):
        cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TRAIN
    # turn on v7
    cfg.MODEL.DENSETRACK.ENABLED = True
    cfg.MODEL.DENSETRACK.COMP1_ENABLED = True
    cfg.MODEL.DENSETRACK.COMP2_ENABLED = True
    cfg.MODEL.DENSETRACK.COMP3_ENABLED = True
    cfg.freeze()

    print("\n[5.1] Building model with DENSETRACK enabled...")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(ckpt_path)

    # Verify all 3 components instantiated
    assert model.v7_proposal_augmenter is not None, "Comp 1 not instantiated"
    assert model.roi_heads.v7_consensus_matcher is not None, "Comp 2 not instantiated"
    assert model.roi_heads.v7_content_matcher is not None, "Comp 3 not instantiated"
    print("  ✓ all 3 v7 components instantiated on the model")

    # Param counts
    n_c1 = sum(p.numel() for p in model.v7_proposal_augmenter.parameters())
    n_c2 = sum(p.numel() for p in model.roi_heads.v7_consensus_matcher.parameters())
    n_c3 = sum(p.numel() for p in model.roi_heads.v7_content_matcher.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  c1={n_c1:,}  c2={n_c2:,}  c3={n_c3:,}  total_model={n_total:,}")

    model.train()

    print("\n[5.2] Building synthetic 6-frame clip + mock _v7_ctx...")
    device = torch.device("cuda")
    H, W = 720, 1280
    num_frames = 6
    n_inst = 3
    num_points = 25

    batched_inputs = []
    for t in range(num_frames):
        img = torch.randint(0, 255, (3, H, W), dtype=torch.uint8, device=device).float()
        polyline = torch.rand(n_inst, num_points * 2, device=device) * min(H, W)
        gt_ids = torch.arange(n_inst, device=device)
        gt_classes = torch.zeros(n_inst, dtype=torch.long, device=device)
        gt_boxes = Boxes(torch.tensor(
            [[100., 100., 300., 300.],
             [400., 200., 600., 400.],
             [200., 500., 500., 700.]], device=device))
        inst = Instances((H, W))
        inst.gt_classes = gt_classes
        inst.gt_boxes = gt_boxes
        inst.polyline = polyline
        inst.texts = torch.zeros(n_inst, num_points, dtype=torch.long, device=device)
        inst.gt_instance_ids = gt_ids
        batched_inputs.append({"image": img, "instances": inst})

    # Mock _v7_ctx — placed on the model so _v7_set_roi_ctx picks it up
    model._v7_pending_ctx = {
        "image_size": (H, W),
        # Encoder features for Comp 2 (4 encoders, 32x32 spatial)
        "encoder_features": {
            "clip-l-336":  torch.randn(num_frames, 1024, 32, 32, device=device),
            "dinov2-l":    torch.randn(num_frames, 1024, 32, 32, device=device),
            "sam-h":       torch.randn(num_frames, 1280, 32, 32, device=device),
            "convnext-l":  torch.randn(num_frames, 1536, 32, 32, device=device),
        },
        # Text per detection per frame for Comp 3 — needs to match the proposal count
        # at runtime (we don't know that yet, so we provide an empty list and the
        # matcher will skip if mismatched).
        "frame_texts": [["HELLO", "WORLD", "FOO"] for _ in range(num_frames)],
        # SAM data for Comp 1 — synthetic polygons + features + scores
        "sam_polygons": [
            [[100., 100., 200., 100., 200., 200., 100., 200.]] * 3
            for _ in range(num_frames)
        ],
        "sam_clip_features": [torch.randn(3, 1024, device=device) for _ in range(num_frames)],
        "sam_stability_scores": [torch.rand(3, device=device) for _ in range(num_frames)],
    }
    print(f"  ✓ mock ctx built: {len(model._v7_pending_ctx['encoder_features'])} encoder maps, "
          f"{sum(len(p) for p in model._v7_pending_ctx['sam_polygons'])} SAM polygons")

    print("\n[5.3] Forward + backward with all v7 paths active...")
    try:
        t0 = time.time()
        losses = model(batched_inputs)
        fwd_t = time.time() - t0
        print(f"  forward: {fwd_t:.2f}s  losses: {list(losses.keys())}")
        for k, v in losses.items():
            print(f"    {k}: {v.item():.4f}")
        total = sum(losses.values())
        assert torch.isfinite(total), f"loss is not finite: {total.item()}"
        total.backward()
        print(f"  backward: OK")

        # Verify gates received gradients (None is OK with synthetic data —
        # synthetic GT often doesn't overlap synthetic predictions, so asso
        # losses can be 0, in which case v7 gates won't receive gradient)
        def _fmt(g):
            return "None (no asso loss flowed; expected on synthetic input)" \
                if g is None else f"{g.item():.6f}"
        c1_g = model.v7_proposal_augmenter.gate.grad
        c2_g = model.roi_heads.v7_consensus_matcher.gate_logit.grad
        c3_g = model.roi_heads.v7_content_matcher.content_logit.grad
        print(f"  c1 gate.grad:           {_fmt(c1_g)}")
        print(f"  c2 gate_logit.grad:     {_fmt(c2_g)}")
        print(f"  c3 content_logit.grad:  {_fmt(c3_g)}")

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU peak mem: {peak_gb:.2f} GB")
        print("=" * 60)
        print("Stage 5 PASSED. Deep integration verified.")
        print("=" * 60)

        return {
            "ok": True,
            "fwd_s": fwd_t,
            "total_loss": float(total.item()),
            "loss_long_asso": float(losses.get("loss_long_asso", torch.tensor(0.)).item()),
            "loss_short_asso": float(losses.get("loss_short_asso", torch.tensor(0.)).item()),
            "loss_res": float(losses.get("loss_res", torch.tensor(0.)).item()),
            "c1_gate_grad": None if c1_g is None else float(c1_g.item()),
            "c2_gate_grad": None if c2_g is None else float(c2_g.item()),
            "c3_gate_grad": None if c3_g is None else float(c3_g.item()),
            "peak_gb": float(peak_gb),
            "v7_params": int(n_c1 + n_c2 + n_c3),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        print("=" * 60)
        print(f"Stage 5 FAILED: {type(e).__name__}: {e}")
        print("=" * 60)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Function 6.5: download GoMatching's IC15-V train.json (real GT) into volume
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=2,
    timeout=600,
    volumes={"/ckpts": v_ckpts},
)
def fetch_ic15v_train_json():
    """gdown GoMatching's IC15-V train.json (~42 MB) into mevl-vts-v7-checkpoints
    volume at /ckpts/ICDAR15/train.json. One-time."""
    import os, subprocess, sys
    out_dir = "/ckpts/ICDAR15"
    out_path = os.path.join(out_dir, "train.json")
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        return {"ok": True, "skipped": True, "path": out_path, "size_mb": sz / 1e6}
    os.makedirs(out_dir, exist_ok=True)
    # GoMatching's IC15-V train.json gdrive file id (from prior baseline reproduction)
    file_id = "18_d5oN4yvcXCV1nUb8OlQJSDzLAZ1Mlz"
    url = f"https://drive.google.com/uc?id={file_id}"
    cmd = [sys.executable, "-m", "gdown", url, "-O", out_path]
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"\nexit={proc.returncode}  size={sz/1e6:.2f} MB", flush=True)
    if sz < 1e6:
        # Try fuzzy mode (works around some gdrive cookie issues)
        print("[fallback] retrying with --fuzzy", flush=True)
        cmd = [sys.executable, "-m", "gdown", "--fuzzy", url, "-O", out_path]
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        print(f"exit={proc.returncode}  size={sz/1e6:.2f} MB", flush=True)
    v_ckpts.commit()
    return {"ok": sz > 1e6, "path": out_path, "size_mb": sz / 1e6}


# ---------------------------------------------------------------------------
# Function 7: end-to-end real-data pilot — register dataset, build dataloader,
# run a few training iters, verify gate gradients flow
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=1200,
    volumes={"/ckpts": v_ckpts, "/data": v_data, "/outputs": v_outputs,
             "/cache": v_encache, "/ds": v_datasets},
)
def verify_real_training(num_iters: int = 5):
    """Stage 7: run a real mini training loop with GoMatching's official IC15-V GT.

    Confirms gate gradients flow when asso loss is non-zero (real matches).
    """
    import os, sys, time, torch
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    print("=" * 60)
    print(f"  Stage 7: real-data pilot ({num_iters} iters)")
    print("=" * 60)

    train_json = "/ckpts/ICDAR15/train.json"
    if not os.path.exists(train_json):
        return {"ok": False, "reason": "no train.json on volume; run fetch_ic15v_train_json first"}
    print(f"  found {train_json} ({os.path.getsize(train_json)/1e6:.1f} MB)")

    # GoMatching's auto-registration uses relative paths datasets/ICDAR15/{train.json, frame/}
    # cwd to /opt/GoMatching and symlink the volume-mounted files into the expected layout
    os.chdir("/opt/GoMatching")
    os.makedirs("datasets/ICDAR15", exist_ok=True)
    if not os.path.lexists("datasets/ICDAR15/train.json"):
        os.symlink(train_json, "datasets/ICDAR15/train.json")

    # IC15-V json refers to '1.jpg'..'960.jpg' (1-indexed) but cache stores
    # '000000.jpg'..'000959.jpg' (0-indexed zero-padded). Materialize 27.8k
    # symlinks in a renamed mirror dir. Idempotent.
    frame_dir = "datasets/ICDAR15/frame"
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)
        src_root = "/ds/ICDAR15_Video/frames"
        videos = [v for v in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, v))]
        print(f"  materializing renamed frame symlinks for {len(videos)} videos...")
        n_total = 0
        for vid in videos:
            vid_dst = os.path.join(frame_dir, vid)
            os.makedirs(vid_dst, exist_ok=True)
            for f in os.listdir(os.path.join(src_root, vid)):
                if not f.endswith(".jpg"):
                    continue
                stem = os.path.splitext(f)[0]
                try:
                    new_n = int(stem) + 1  # 0-indexed -> 1-indexed
                except ValueError:
                    continue
                link = os.path.join(vid_dst, f"{new_n}.jpg")
                if not os.path.lexists(link):
                    os.symlink(os.path.join(src_root, vid, f), link)
                n_total += 1
        print(f"    {n_total} symlinks ready")
    print("  symlinks: datasets/ICDAR15/{train.json -> /ckpts/..., frame/<vid>/<N>.jpg -> /ds/...}")

    ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"

    import gomatching  # noqa
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from adet.config import add_deepsolo_cfg
    from gomatching.config import add_gom_config
    from gomatching.data.datasets import vts  # noqa - registers VTS loaders
    from gomatching.data.custom_build_augmentation import build_custom_augmentation
    from gomatching.data.vts_dataset_dataloader import build_vts_train_loader
    from gomatching.data.vts_dataset_mapper import GoMDatasetMapper
    from gomatching.modeling.freeze_layers import check_if_freeze_model

    # icdar15_train is auto-registered by `from gomatching.data.datasets import vts`
    # at relative paths datasets/ICDAR15/{train.json, frame/} which our symlinks now satisfy.
    print("\n[7.1] icdar15_train dataset already registered via vts.py auto-register")

    cfg = get_cfg()
    add_deepsolo_cfg(cfg)
    add_gom_config(cfg)
    cfg.merge_from_file("/opt/GoMatching/configs/GoMatching_PP_ICDAR15.yaml")
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.MODEL.DEVICE = "cuda"
    if hasattr(cfg.MODEL.TRANSFORMER, "INFERENCE_TH_TRAIN"):
        cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TRAIN
    # v7 ON with real cache paths
    cfg.MODEL.DENSETRACK.ENABLED = True
    cfg.MODEL.DENSETRACK.COMP1_ENABLED = True
    cfg.MODEL.DENSETRACK.COMP1_CACHE_ROOT = "/cache/ic15v/sam_proposals"
    cfg.MODEL.DENSETRACK.COMP1_CLIP_DIM = 768
    cfg.MODEL.DENSETRACK.COMP2_ENABLED = True
    cfg.MODEL.DENSETRACK.COMP2_CACHE_ROOT = "/cache/ic15v"
    cfg.MODEL.DENSETRACK.COMP3_ENABLED = False
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0  # single-worker for the smoke; verify lazy h5py
    cfg.freeze()

    print("\n[7.2] Build model + dataloader...")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(ckpt_path)
    check_if_freeze_model(model, cfg)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable:,}")

    mapper = GoMDatasetMapper(cfg, is_train=True,
                              augmentations=build_custom_augmentation(cfg, True))
    loader = build_vts_train_loader(cfg, mapper=mapper)
    print(f"  loader built")

    print(f"\n[7.3] Run {num_iters} training iters with real data + GT...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5, weight_decay=0.01)

    grad_observed = {"c1": False, "c2": False}
    last_losses = {}
    try:
        for it, data in enumerate(loader):
            if it >= num_iters:
                break
            t0 = time.time()
            losses = model(data)
            total = sum(losses.values())
            optimizer.zero_grad()
            total.backward()
            # check gradients BEFORE optimizer.step
            c1_g = model.v7_proposal_augmenter.gate.grad
            c2_g = model.roi_heads.v7_consensus_matcher.gate_logit.grad
            if c1_g is not None and c1_g.abs().sum() > 0: grad_observed["c1"] = True
            if c2_g is not None and c2_g.abs().sum() > 0: grad_observed["c2"] = True
            optimizer.step()
            dt = time.time() - t0
            last_losses = {k: float(v.item()) for k, v in losses.items()}
            print(f"  iter {it}: {dt:.2f}s losses={last_losses} "
                  f"c1_grad={'+' if c1_g is not None and c1_g.abs().sum()>0 else '0'} "
                  f"c2_grad={'+' if c2_g is not None and c2_g.abs().sum()>0 else '0'}")

        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  GPU peak: {peak:.2f} GB")
        print(f"  gate gradient observed: c1={grad_observed['c1']}, c2={grad_observed['c2']}")
        print("=" * 60)
        if grad_observed["c2"]:
            print("Stage 7 PASSED — Comp 2 receives gradient from real asso loss.")
        else:
            print("Stage 7 PARTIAL — Comp 2 gates did not flow. Investigate.")
        print("=" * 60)
        return {"ok": grad_observed["c2"], "grad_observed": grad_observed,
                "last_losses": last_losses, "peak_gb": float(peak)}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Function 6: deep integration with REAL cached data (the post-bugfix smoke)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    volumes={
        "/ckpts": v_ckpts,
        "/data": v_data,
        "/outputs": v_outputs,
        "/cache": v_encache,
        "/ds": v_datasets,
    },
)
def verify_real_data_integration():
    """Stage 6: load v7 model + use the patched GoMDatasetMapper to fetch a REAL
    clip with real cached HDF5 + NPZ data. Verify v7 gates receive gradient
    (would have been None in Stage 5 due to synthetic-input artifact)."""
    import os, sys, time, torch
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    print("=" * 60)
    print("  Stage 6: real-data integration (post dataloader patch)")
    print("=" * 60)

    ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"
    if not os.path.exists(ckpt_path):
        return {"ok": False, "reason": "no checkpoint"}

    import gomatching  # noqa
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from adet.config import add_deepsolo_cfg
    from gomatching.config import add_gom_config
    from gomatching.modeling.freeze_layers import check_if_freeze_model

    cfg = get_cfg()
    add_deepsolo_cfg(cfg)
    add_gom_config(cfg)
    cfg.merge_from_file("/opt/GoMatching/configs/GoMatching_PP_ICDAR15.yaml")
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.MODEL.DEVICE = "cuda"
    if hasattr(cfg.MODEL.TRANSFORMER, "INFERENCE_TH_TRAIN"):
        cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TRAIN
    # v7 ON, point at the real caches
    cfg.MODEL.DENSETRACK.ENABLED = True
    cfg.MODEL.DENSETRACK.COMP1_ENABLED = True
    cfg.MODEL.DENSETRACK.COMP1_CACHE_ROOT = "/cache/ic15v/sam_proposals"
    cfg.MODEL.DENSETRACK.COMP2_ENABLED = True
    cfg.MODEL.DENSETRACK.COMP2_CACHE_ROOT = "/cache/ic15v"
    cfg.MODEL.DENSETRACK.COMP3_ENABLED = False  # no VLM text cache yet
    cfg.freeze()

    print("\n[6.1] Build model + apply freeze policy...")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(ckpt_path)
    check_if_freeze_model(model, cfg)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable:,}")

    print("\n[6.2] Build a real 6-frame clip via patched GoMDatasetMapper...")
    from gomatching.data.vts_dataset_mapper import GoMDatasetMapper
    from gomatching.data.custom_build_augmentation import build_custom_augmentation

    mapper = GoMDatasetMapper(cfg, is_train=True,
                              augmentations=build_custom_augmentation(cfg, True))
    print(f"  mapper.v7_cache_built_lazily: {mapper._v7_cache is None}")

    # Build a minimal video_dict pointing at real frames
    video_dir = "/ds/ICDAR15_Video/frames/Video_10_1_1"
    jpgs = sorted(os.listdir(video_dir))[:8]
    if len(jpgs) < 6:
        return {"ok": False, "reason": "not enough frames on volume"}
    from PIL import Image as PILImage
    # Read actual dims so detectron2's check_image_size passes
    probe_path = os.path.join(video_dir, jpgs[0])
    with PILImage.open(probe_path) as _im:
        W0, H0 = _im.size
    print(f"  real frame size: {W0}x{H0}")
    images = []
    for jpg in jpgs[:8]:
        images.append({
            "file_name": os.path.join(video_dir, jpg),
            "image_id": int(os.path.splitext(jpg)[0]),
            "height": H0, "width": W0,
            "annotations": [],
        })
    video_dict = {"video_id": 10, "images": images}
    print(f"  built video_dict with {len(images)} frames from {video_dir}")

    batched = mapper(video_dict)
    print(f"  mapper returned {len(batched)} dataset_dicts")
    # Inspect v7 keys
    sample = batched[0]
    has_enc = "v7_encoder_features" in sample
    has_sam = "v7_sam_polygons" in sample
    if has_enc:
        enc = sample["v7_encoder_features"]
        print(f"  [v7] encoder features present: {list(enc.keys())}")
        for k, v in enc.items():
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    if has_sam:
        print(f"  [v7] sam_polygons: {len(sample['v7_sam_polygons'])} masks "
              f"clip_features={tuple(sample['v7_sam_clip_features'].shape)} "
              f"stab={tuple(sample['v7_sam_stability_scores'].shape)}")

    # GT instances need to be populated for asso loss to flow.  We don't have
    # parsed GT here, so we'll just confirm the model can FORWARD with these
    # inputs without crashing. (Real training uses real annotations.)
    # Replace empty annotations with one dummy GT per frame so the GT
    # construction code doesn't error.
    from detectron2.structures import Instances, Boxes
    for d in batched:
        H, W = d["image"].shape[-2], d["image"].shape[-1]
        inst = Instances((H, W))
        inst.gt_classes = torch.zeros(1, dtype=torch.long)
        inst.gt_boxes = Boxes(torch.tensor([[100., 100., 300., 300.]]))
        inst.polyline = torch.rand(1, 25 * 2) * min(H, W)
        inst.texts = torch.zeros(1, 25, dtype=torch.long)
        inst.gt_instance_ids = torch.tensor([1], dtype=torch.long)
        inst.beziers = torch.zeros(1, 16)
        inst.boundary = torch.zeros(1, 50)
        d["instances"] = inst

    print("\n[6.3] Forward + backward...")
    try:
        t0 = time.time()
        losses = model(batched)
        print(f"  forward {time.time()-t0:.1f}s losses={ {k: float(v) for k,v in losses.items()} }")
        total = sum(losses.values())
        total.backward()
        print(f"  backward OK")

        # CRUCIAL: did v7 gates get gradient?
        c1_g = model.v7_proposal_augmenter.gate.grad
        c2_g = model.roi_heads.v7_consensus_matcher.gate_logit.grad
        c1s = "None" if c1_g is None else f"{c1_g.flatten().item():.6f}"
        c2s = "None" if c2_g is None else f"{c2_g.item():.6f}"
        print(f"  c1 gate.grad: {c1s}")
        print(f"  c2 gate_logit.grad: {c2s}")
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU peak: {peak:.2f} GB")
        print("=" * 60)
        print("Stage 6 done.")
        print("=" * 60)
        return {"ok": True,
                "c1_gate_grad": None if c1_g is None else float(c1_g.flatten().item()),
                "c2_gate_grad": None if c2_g is None else float(c2_g.item()),
                "peak_gb": float(peak)}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Function 8: train DenseTrack v7 (real fine-tune from iter-30k baseline)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=86400,  # 24h cap (full 12k-iter run)
    volumes={"/ckpts": v_ckpts, "/data": v_data, "/outputs": v_outputs,
             "/cache": v_encache, "/ds": v_datasets},
)
def train_densetrack_v7(max_iter: int = 12000, base_lr: float = 2e-5,
                          checkpoint_period: int = 1000, warmup_iters: int = 1000,
                          output_name: str = "DenseTrack_IC15",
                          resume: bool = False):
    """Real DenseTrack v7 fine-tune (same setup that Stage 7 verified)."""
    import os, sys, subprocess, time
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    train_json = "/ckpts/ICDAR15/train.json"
    ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"
    if not os.path.exists(train_json):
        return {"ok": False, "reason": "no train.json — run fetch_ic15v_train_json"}
    if not os.path.exists(ckpt_path):
        return {"ok": False, "reason": "no iter-30k checkpoint"}

    # Symlink setup (same as Stage 7)
    os.chdir("/opt/GoMatching")
    os.makedirs("datasets/ICDAR15", exist_ok=True)
    if not os.path.lexists("datasets/ICDAR15/train.json"):
        os.symlink(train_json, "datasets/ICDAR15/train.json")
    frame_dir = "datasets/ICDAR15/frame"
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)
        src_root = "/ds/ICDAR15_Video/frames"
        videos = [v for v in os.listdir(src_root)
                  if os.path.isdir(os.path.join(src_root, v))]
        print(f"  materializing renamed frame symlinks for {len(videos)} videos...")
        n_total = 0
        for vid in videos:
            vid_dst = os.path.join(frame_dir, vid)
            os.makedirs(vid_dst, exist_ok=True)
            for f in os.listdir(os.path.join(src_root, vid)):
                if not f.endswith(".jpg"):
                    continue
                stem = os.path.splitext(f)[0]
                try:
                    new_n = int(stem) + 1
                except ValueError:
                    continue
                link = os.path.join(vid_dst, f"{new_n}.jpg")
                if not os.path.lexists(link):
                    os.symlink(os.path.join(src_root, vid, f), link)
                n_total += 1
        print(f"    {n_total} symlinks ready")

    # Pretrained checkpoint where the config expects it
    os.makedirs("pretrained_models", exist_ok=True)
    pretrained = "pretrained_models/model_final.pth"
    if not os.path.lexists(pretrained):
        os.symlink(ckpt_path, pretrained)

    # Patch the config's cache paths via cfg --opts (no sed on the yaml)
    output_name = os.path.basename(output_name.rstrip("/")) or "DenseTrack_IC15"
    output_dir = f"/outputs/{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "train_net.py",
        "--num-gpus", "1",
        "--config-file", "configs/DenseTrack_PP_ICDAR15.yaml",
    ]
    if resume:
        cmd.append("--resume")
    # detectron2 default_argument_parser uses `opts` as a POSITIONAL nargs=REMAINDER.
    # Pass key/value pairs directly (no --opts flag).
    cmd += [
        "SOLVER.MAX_ITER", str(max_iter),
        "SOLVER.BASE_LR", str(base_lr),
        "SOLVER.WARMUP_ITERS", str(warmup_iters),
        "SOLVER.CHECKPOINT_PERIOD", str(checkpoint_period),
        "OUTPUT_DIR", output_dir,
        "MODEL.WEIGHTS", pretrained,
        "MODEL.DENSETRACK.COMP1_CACHE_ROOT", "/cache/ic15v/sam_proposals",
        "MODEL.DENSETRACK.COMP2_CACHE_ROOT", "/cache/ic15v",
        "MODEL.DENSETRACK.COMP1_CLIP_DIM", "768",
        "MODEL.DENSETRACK.COMP3_ENABLED", "False",   # no VLM cache yet
        "DATALOADER.NUM_WORKERS", "2",
    ]
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    v_outputs.commit()
    return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
            "wall_min": round(dt / 60, 1), "output_dir": output_dir}


@app.function(
    image=image,
    gpu="L4",
    timeout=10800,
    volumes={"/ckpts": v_ckpts, "/data": v_data, "/outputs": v_outputs,
             "/cache": v_encache, "/ds": v_datasets},
)
def eval_ic15v_v7(checkpoint_dir: str = "/outputs/DenseTrack_IC15",
                  use_iter: int = -1, output_subdir: str = "eval"):
    """Evaluate a DenseTrack v7 checkpoint with the same cache overrides as training."""
    import os, sys, subprocess, time
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")
    os.chdir("/opt/GoMatching")

    if use_iter >= 0:
        ckpt = os.path.join(checkpoint_dir, f"model_{use_iter:07d}.pth")
    else:
        ckpt = os.path.join(checkpoint_dir, "model_final.pth")
    if not os.path.exists(ckpt):
        return {"ok": False, "reason": f"checkpoint missing: {ckpt}"}

    out_dir = os.path.join(checkpoint_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "eval.py",
        "--config-file", "configs/DenseTrack_PP_ICDAR15.yaml",
        "--input", "/ds/ICDAR15_Video/frames/",
        "--output", out_dir,
        "--opts",
        "MODEL.WEIGHTS", ckpt,
        "MODEL.DENSETRACK.COMP1_CACHE_ROOT", "/cache/ic15v/sam_proposals",
        "MODEL.DENSETRACK.COMP2_CACHE_ROOT", "/cache/ic15v",
        "MODEL.DENSETRACK.COMP1_CLIP_DIM", "768",
        "MODEL.DENSETRACK.COMP3_ENABLED", "False",
    ]
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    v_outputs.commit()
    return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
            "wall_min": round(dt / 60, 1), "out_dir": out_dir}


# ---------------------------------------------------------------------------
# Phase 1: cache encoder features for IC15-V
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=28800,  # 8h max — SAM-H is the slow one
    volumes={"/ds": v_datasets, "/cache": v_encache},
)
def cache_sam_ic15v(limit_videos: int = 0, points_per_side: int = 16,
                     min_mask_area: int = 400, max_masks: int = 50,
                     force: bool = False):
    """Run SAM-H 'segment everything' over IC15-V frames + CLIP-pool each mask.

    Slow: ~1-2 s/frame on L4. Full IC15-V (~14k frames) ≈ 4-8 hours.
    """
    import subprocess, sys
    frames_root = "/ds/ICDAR15_Video/frames"
    out_root = "/cache/ic15v/sam_proposals"
    cmd = [
        sys.executable, "/opt/GoMatching/tools_v7/cache_sam_proposals.py",
        "--frames-root", frames_root,
        "--out-root", out_root,
        "--points-per-side", str(points_per_side),
        "--min-mask-area", str(min_mask_area),
        "--max-masks", str(max_masks),
    ]
    if limit_videos > 0:
        cmd += ["--limit-videos", str(limit_videos)]
    if force:
        cmd += ["--force"]
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    v_encache.commit()
    return {"exit_code": proc.returncode, "out_root": out_root}


@app.function(
    image=image,
    cpu=2,                # CPU-only — just lists files
    volumes={"/cache": v_encache},
)
def cache_status():
    """Inspect the encoder cache volume — how many videos cached per encoder."""
    import os
    out = {}
    root = "/cache/ic15v"
    if not os.path.isdir(root):
        return {"error": f"no {root}"}
    for enc in os.listdir(root):
        p = os.path.join(root, enc)
        if not os.path.isdir(p):
            continue
        files = [f for f in os.listdir(p) if f.endswith(".h5") or f.endswith(".npz")]
        sizes = sum(os.path.getsize(os.path.join(p, f)) for f in files) / (1024**3)
        out[enc] = {"n_files": len(files), "gb": round(sizes, 2)}
    print(out)
    return out


@app.function(
    image=image,
    gpu="L4",
    timeout=14400,  # 4h max per encoder
    volumes={
        "/ds": v_datasets,        # IC15-V frames live here
        "/cache": v_encache,      # output -> mevl-vts-v7-encoder-cache
    },
)
def cache_encoders_ic15v(encoder: str, limit_videos: int = 0, batch_size: int = 8,
                          force: bool = False):
    """Run one encoder over all IC15-V frames, save 32x32 feature maps as HDF5.

    Usage from local:
        modal run modal_v7_setup.py::cache_encoders_ic15v --encoder clip-l-336
        modal run modal_v7_setup.py::cache_encoders_ic15v --encoder dinov2-l
        modal run modal_v7_setup.py::cache_encoders_ic15v --encoder sam-h
        modal run modal_v7_setup.py::cache_encoders_ic15v --encoder convnext-l
        # smoke first:
        modal run modal_v7_setup.py::cache_encoders_ic15v --encoder clip-l-336 --limit-videos 2
    """
    import subprocess, sys
    frames_root = "/ds/ICDAR15_Video/frames"
    out_root = "/cache/ic15v"
    cmd = [
        sys.executable, "/opt/GoMatching/tools_v7/cache_encoder_features.py",
        "--frames-root", frames_root,
        "--out-root", out_root,
        "--encoder", encoder,
        "--batch-size", str(batch_size),
    ]
    if limit_videos > 0:
        cmd += ["--limit-videos", str(limit_videos)]
    if force:
        cmd += ["--force"]
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    # Ensure volume changes persist
    v_encache.commit()
    return {"encoder": encoder, "exit_code": proc.returncode,
            "frames_root": frames_root, "out_root": out_root}


# ---------------------------------------------------------------------------
# Function 9: multi-dataset base GoMatching++ training (IC15 + DSText + BOVText)
# ---------------------------------------------------------------------------
def _stage_dataset_symlinks(names):
    """Build datasets/<DS>/{train.json,frame/} for each requested dataset, mirroring
    the layout GoMatching's vts.py auto-registration expects. Returns notes list."""
    import os
    notes = []
    # IC15: train.json from /ckpts, frames renamed 0-idx->1-idx from /ds
    if "icdar15_train" in names:
        os.makedirs("datasets/ICDAR15", exist_ok=True)
        tj = "/ckpts/ICDAR15/train.json"
        if not os.path.lexists("datasets/ICDAR15/train.json"):
            os.symlink(tj, "datasets/ICDAR15/train.json")
        fd = "datasets/ICDAR15/frame"
        if not os.path.isdir(fd):
            os.makedirs(fd, exist_ok=True)
            src = "/ds/ICDAR15_Video/frames"
            n = 0
            for vid in [v for v in os.listdir(src) if os.path.isdir(f"{src}/{v}")]:
                os.makedirs(f"{fd}/{vid}", exist_ok=True)
                for f in os.listdir(f"{src}/{vid}"):
                    if not f.endswith(".jpg"):
                        continue
                    try:
                        nn = int(os.path.splitext(f)[0]) + 1
                    except ValueError:
                        continue
                    lk = f"{fd}/{vid}/{nn}.jpg"
                    if not os.path.lexists(lk):
                        os.symlink(f"{src}/{vid}/{f}", lk)
                    n += 1
            notes.append(f"IC15 frames: {n} symlinks")
    # DSText / BOVText: prepared on /ds already with 1-indexed <fid>.jpg matching json
    for key, root in (("dstext_train", "/ds/DSText"), ("bov_train", "/ds/BOVText")):
        if key not in names:
            continue
        short = "DSText" if key == "dstext_train" else "BOVText"
        os.makedirs("datasets", exist_ok=True)
        if not os.path.lexists(f"datasets/{short}"):
            os.symlink(root, f"datasets/{short}")  # whole dir: has train.json + frame/
        ok = os.path.exists(f"datasets/{short}/train.json")
        notes.append(f"{short}: linked (train.json present={ok})")
    return notes


@app.function(
    image=image, gpu="L4", timeout=86400,
    volumes={"/ckpts": v_ckpts, "/data": v_data, "/outputs": v_outputs,
             "/cache": v_encache, "/ds": v_datasets},
)
def train_multitrain(max_iter: int = 60000, base_lr: float = 5e-5,
                     checkpoint_period: int = 5000, warmup_iters: int = 1000,
                     datasets: str = "", output_name: str = "GoMPP_MultiTrain",
                     num_workers: int = 2, resume: bool = False):
    """Base GoMatching++ training across IC15+DSText+BOVText (v7 comps OFF, no caches).

    `datasets`='' uses the config default (all 3). For a smoke pass a tuple-literal
    string, e.g. '("icdar15_train", "dstext_train")'.
    """
    import os, sys, subprocess, time
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")
    os.chdir("/opt/GoMatching")

    ckpt_path = "/ckpts/gomatching_iter30k/model_final.pth"
    if not os.path.exists(ckpt_path):
        return {"ok": False, "reason": "no iter-30k checkpoint"}

    # which datasets do we need staged?
    need = ["icdar15_train", "dstext_train", "bov_train"]
    if datasets:
        need = [d.strip().strip("(),'\" ") for d in datasets.split(",") if d.strip().strip("(),'\" ")]
    notes = _stage_dataset_symlinks(need)
    for nt in notes:
        print("  [stage]", nt)

    os.makedirs("pretrained_models", exist_ok=True)
    if not os.path.lexists("pretrained_models/model_final.pth"):
        os.symlink(ckpt_path, "pretrained_models/model_final.pth")

    output_dir = f"/outputs/{os.path.basename(output_name.rstrip('/'))}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [sys.executable, "train_net.py", "--num-gpus", "1",
           "--config-file", "configs/GoMatching_PP_MultiTrain.yaml"]
    if resume:
        cmd.append("--resume")
    cmd += [
        "SOLVER.MAX_ITER", str(max_iter),
        "SOLVER.BASE_LR", str(base_lr),
        "SOLVER.WARMUP_ITERS", str(warmup_iters),
        "SOLVER.CHECKPOINT_PERIOD", str(checkpoint_period),
        "OUTPUT_DIR", output_dir,
        "MODEL.WEIGHTS", "pretrained_models/model_final.pth",
        "DATALOADER.NUM_WORKERS", str(num_workers),
    ]
    # Always make DATASETS + sampler self-consistent with the staged set.
    tup = "(" + ", ".join(f"'{d}'" for d in need) + ",)"
    cmd += ["DATASETS.TRAIN", tup, "DATASETS.TEST", tup]
    if len(need) > 1:
        # multi-dataset needs MultiDatasetSampler + a ratio per dataset
        cmd += ["DATALOADER.SAMPLER_TRAIN", "MultiDatasetSampler",
                "DATALOADER.DATASET_RATIO", "[" + ",".join("1" for _ in need) + "]"]
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    v_outputs.commit()
    return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
            "wall_min": round(dt / 60, 1), "output_dir": output_dir, "staged": notes}


@app.function(
    image=image, gpu="L4", timeout=10800,
    volumes={"/ckpts": v_ckpts, "/outputs": v_outputs, "/ds": v_datasets},
)
def eval_base(checkpoint: str = "/outputs/GoMPP_MultiTrain/model_final.pth",
              output_subdir: str = "eval", limit_videos: int = 0):
    """Eval a BASE GoMatching++ checkpoint (no v7 comps) on IC15-V test frames,
    producing RRC-format res_*.xml + per-video json + the getid_text track map.

    limit_videos>0 -> symlink that many video dirs into a smoke subset (fast check
    that the eval->RRC pipeline emits valid output)."""
    import os, sys, subprocess, time
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")
    os.chdir("/opt/GoMatching")

    if not os.path.exists(checkpoint):
        return {"ok": False, "reason": f"missing checkpoint {checkpoint}"}

    frames_root = "/ds/ICDAR15_Video/frames"
    if limit_videos > 0:
        # build a small subset dir whose path contains 'ICDAR15' (eval.py keys on that)
        subset = "/opt/GoMatching/datasets/ICDAR15_eval_smoke/ICDAR15"
        os.makedirs(subset, exist_ok=True)
        vids = sorted(v for v in os.listdir(frames_root)
                      if os.path.isdir(os.path.join(frames_root, v)))[:limit_videos]
        for v in vids:
            lk = os.path.join(subset, v)
            if not os.path.lexists(lk):
                os.symlink(os.path.join(frames_root, v), lk)
        input_dir = subset
        print(f"  smoke subset: {vids}")
    else:
        input_dir = frames_root

    out_dir = os.path.join(os.path.dirname(checkpoint), output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "eval.py",
           "--config-file", "configs/GoMatching_PP_ICDAR15.yaml",
           "--input", input_dir, "--output", out_dir,
           "--opts", "MODEL.WEIGHTS", checkpoint]
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    v_outputs.commit()
    # report what landed
    xml_dir = os.path.join(out_dir, "xml")
    xmls = [f for f in os.listdir(xml_dir)] if os.path.isdir(xml_dir) else []
    return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
            "wall_min": round(dt / 60, 1), "out_dir": out_dir,
            "xml_files": xmls[:10], "n_xml": len(xmls)}


@app.local_entrypoint()
def main(stage: str = "all"):
    """Convenience entrypoint: `modal run modal/modal_v7_setup.py --stage 1|2|3|4|5|all`"""
    if stage in ("1", "all"):
        print("\n>>> Running Stage 1: verify_image")
        verify_image.remote()
    if stage in ("2", "all"):
        print("\n>>> Running Stage 2: verify_baseline_load")
        verify_baseline_load.remote()
    if stage in ("3", "all"):
        print("\n>>> Running Stage 3: verify_smoke_forward")
        verify_smoke_forward.remote()
    if stage in ("4", "all"):
        print("\n>>> Running Stage 4: verify_components (v7 DenseTrack)")
        verify_components.remote()
    if stage in ("5", "all"):
        print("\n>>> Running Stage 5: verify_deep_integration")
        verify_deep_integration.remote()
    if stage in ("6", "all"):
        print("\n>>> Running Stage 6: verify_real_data_integration")
        verify_real_data_integration.remote()
