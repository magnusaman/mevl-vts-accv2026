"""Modal app for Phase 2: train DenseTrack v7 from iter-30k baseline.

Loads /opt/GoMatching/configs/DenseTrack_PP_ICDAR15.yaml (which inherits the
GoMatching++ base config and enables all 3 v7 components), warms up from the
iter-30k checkpoint, fine-tunes for ~12k iters at lr=2e-5.

Usage:
    modal deploy modal_v7_train.py
    modal run modal_v7_train.py::train --max-iter 12000

This deliberately imports the SAME image / volumes as modal_v7_setup.py so
deploys are coherent (same code mounted, same caches read).
"""
import os, sys
import modal

# Mirror modal_v7_setup.py's image + volumes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_v7_setup import (
    app, image, v_ckpts, v_data, v_outputs, v_encache, v_datasets,
)


@app.function(
    image=image,
    gpu="L4",
    timeout=86400,  # 24h
    volumes={
        "/ckpts": v_ckpts,
        "/data": v_data,
        "/outputs": v_outputs,
        "/cache": v_encache,
        "/ds": v_datasets,
    },
)
def train(max_iter: int = 12000, config_file: str = "DenseTrack_PP_ICDAR15.yaml",
          resume: bool = False):
    """Run GoMatching/train_net.py with our DenseTrack config.

    Warm-starts from /ckpts/gomatching_iter30k/model_final.pth.

    `resume=True` reloads from /outputs/DenseTrack_IC15/last_checkpoint instead
    (use after a partial run got interrupted).
    """
    import subprocess, sys, os
    os.chdir("/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    # Stage the baseline checkpoint under pretrained_models/ where the config expects it
    src_ckpt = "/ckpts/gomatching_iter30k/model_final.pth"
    pretrained_dir = "/opt/GoMatching/pretrained_models"
    os.makedirs(pretrained_dir, exist_ok=True)
    dst_ckpt = os.path.join(pretrained_dir, "model_final.pth")
    if not os.path.exists(dst_ckpt):
        # Symlink (instant, no copy)
        os.symlink(src_ckpt, dst_ckpt)
        print(f"  symlinked {src_ckpt} -> {dst_ckpt}")

    # Stage caches under datasets/ where the dataloader will find them
    # (config keys MODEL.DENSETRACK.COMP*_CACHE_ROOT all point under /data)

    opts = [
        "SOLVER.MAX_ITER", str(max_iter),
        "OUTPUT_DIR", "/outputs/DenseTrack_IC15",
    ]
    if not resume:
        opts += ["MODEL.WEIGHTS", dst_ckpt]

    cmd = [
        sys.executable, "train_net.py",
        "--num-gpus", "1",
        "--config-file", f"configs/{config_file}",
    ]
    if resume:
        cmd += ["--resume"]
    cmd += opts

    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    v_outputs.commit()
    return {"exit_code": proc.returncode, "output_dir": "/outputs/DenseTrack_IC15"}


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,  # 2h
    volumes={
        "/ckpts": v_ckpts,
        "/data": v_data,
        "/outputs": v_outputs,
        "/cache": v_encache,
        "/ds": v_datasets,
    },
)
def eval_ic15v(checkpoint_dir: str = "/outputs/DenseTrack_IC15",
                use_iter: int = -1, output_subdir: str = "eval"):
    """Run eval.py on IC15-V test split with our trained DenseTrack checkpoint.

    Uses the same eval pipeline GoMatching ships:
        cd /opt/GoMatching && python eval.py --config-file <cfg> ...

    Output XMLs / JSONs go to /outputs/DenseTrack_IC15/eval/
    Build an RRC submission zip after eval finishes (separate step).
    """
    import subprocess, sys, os
    os.chdir("/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching")
    sys.path.insert(0, "/opt/GoMatching/third_party")

    if use_iter >= 0:
        ckpt = os.path.join(checkpoint_dir, f"model_{use_iter:07d}.pth")
    else:
        ckpt = os.path.join(checkpoint_dir, "model_final.pth")
    if not os.path.exists(ckpt):
        return {"error": f"checkpoint missing: {ckpt}"}

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
    proc = subprocess.run(cmd, check=False)
    v_outputs.commit()
    return {"exit_code": proc.returncode, "out_dir": out_dir}


@app.local_entrypoint(name="train_main")
def main(max_iter: int = 12000, resume: bool = False):
    print(f">>> launching DenseTrack v7 training (max_iter={max_iter}, resume={resume})")
    result = train.remote(max_iter=max_iter, resume=resume)
    print(f">>> done: {result}")
