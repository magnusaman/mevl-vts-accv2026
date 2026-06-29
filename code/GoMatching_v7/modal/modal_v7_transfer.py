"""Modal CPU-only app for transferring v7 caches from Modal volume to E2E box.

Why: downloading 175 GB via home connection would take 8-16 hours. Modal's
datacenter → E2E network is much faster (~1-4 hours typical).

Prereq: create a Modal secret containing your E2E SSH private key:
    modal secret create e2e-ssh SSH_PRIVATE_KEY="$(cat ~/.ssh/id_ed25519_e2e_new)"

Usage:
    modal deploy modal_v7_transfer.py
    modal run modal_v7_transfer.py::push_caches_to_e2e
    # or to test on the small SAM cache first:
    modal run modal_v7_transfer.py::push_caches_to_e2e --only sam_proposals

After all transfers succeed, delete the volume to stop storage charges:
    modal volume delete mevl-vts-v7-encoder-cache  # ~$15/mo at 175 GB
"""
import os
import modal

APP_NAME = "mevl-vts-v7-transfer"
app = modal.App(APP_NAME)

# Reference existing volume from the main app
v_encache = modal.Volume.from_name("mevl-vts-v7-encoder-cache", create_if_missing=False)
v_ckpts = modal.Volume.from_name("mevl-vts-v7-checkpoints", create_if_missing=False)

# Slim image with just rsync + ssh client + python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openssh-client", "rsync")
)

ssh_secret = modal.Secret.from_name("e2e-ssh", required_keys=["SSH_PRIVATE_KEY"])


@app.function(
    image=image,
    cpu=4,           # more CPU helps rsync compression
    timeout=21600,   # 6h max (175 GB at ~100 Mbps = 4h)
    volumes={"/cache": v_encache, "/ckpts": v_ckpts},
    secrets=[ssh_secret],
)
def push_caches_to_e2e(
    e2e_host: str = "164.52.193.68",
    e2e_user: str = "root",
    remote_path: str = "/root/data/v7_cache",
    only: str = "",            # if set, only push this subdir (e.g. "sam_proposals")
    dry_run: bool = False,
):
    """Push v7 caches from Modal volume to E2E box via rsync over ssh.

    Default destination on E2E: /root/data/v7_cache/
    Layout there will mirror /cache/ic15v/<encoder>/<video>.h5 etc.

    `--only X` restricts to a single subdir under /cache/ic15v/ — useful for
    a fast sanity check before committing to the full 175 GB push.
    """
    import os, subprocess, sys, time

    # Write SSH key from secret to disk
    key_content = os.environ["SSH_PRIVATE_KEY"]
    key_path = "/tmp/id_e2e"
    with open(key_path, "w") as f:
        f.write(key_content)
        if not key_content.endswith("\n"):
            f.write("\n")
    os.chmod(key_path, 0o600)
    print(f"[setup] SSH key written to {key_path}")

    ssh_opts = ("-i /tmp/id_e2e -o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null -o ServerAliveInterval=60")

    # Ensure remote dir exists
    mk_cmd = f"ssh {ssh_opts} {e2e_user}@{e2e_host} mkdir -p {remote_path}"
    print(f"[setup] $ {mk_cmd}")
    subprocess.run(mk_cmd, shell=True, check=True)

    # Decide source
    if only:
        src = f"/cache/ic15v/{only}/"
        dst = f"{e2e_user}@{e2e_host}:{remote_path}/{only}/"
    else:
        src = "/cache/"
        dst = f"{e2e_user}@{e2e_host}:{remote_path}/"

    rsync_cmd = (
        f"rsync -av --info=progress2 --no-i-r --partial "
        f"-e 'ssh {ssh_opts}' "
        f"{'--dry-run ' if dry_run else ''}"
        f"{src} {dst}"
    )
    print(f"[rsync] $ {rsync_cmd}")
    t0 = time.time()
    proc = subprocess.run(rsync_cmd, shell=True, check=False,
                          stdout=sys.stdout, stderr=sys.stderr)
    dt = time.time() - t0
    print(f"[done] rsync exit={proc.returncode} in {dt/60:.1f} min")

    # Verify byte count on remote
    verify_cmd = f"ssh {ssh_opts} {e2e_user}@{e2e_host} du -sh {remote_path}"
    print(f"[verify] $ {verify_cmd}")
    subprocess.run(verify_cmd, shell=True, check=False)

    return {"exit_code": proc.returncode, "wall_min": round(dt / 60, 1)}


@app.function(
    image=image,
    cpu=2,
    timeout=600,
    volumes={"/cache": v_encache},
    secrets=[ssh_secret],
)
def smoke_ssh_to_e2e(e2e_host: str = "164.52.193.68", e2e_user: str = "root"):
    """Just verify ssh works + remote disk has room. No data transferred."""
    import os, subprocess
    key_content = os.environ["SSH_PRIVATE_KEY"]
    key_path = "/tmp/id_e2e"
    with open(key_path, "w") as f:
        f.write(key_content)
        if not key_content.endswith("\n"):
            f.write("\n")
    os.chmod(key_path, 0o600)
    ssh_opts = ("-i /tmp/id_e2e -o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null")
    cmd = (f"ssh {ssh_opts} {e2e_user}@{e2e_host} "
           f"'echo HOST=$(hostname); echo CWD=$(pwd); df -h /root | tail -1; "
           f"echo CACHE_DIR=/root/data/v7_cache; ls -la /root/data/ 2>/dev/null'")
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, check=False)
    return {"exit_code": proc.returncode}


@app.local_entrypoint()
def main(only: str = "", dry_run: bool = False):
    """`modal run modal_v7_transfer.py` -> push everything.
    `modal run modal_v7_transfer.py --only sam_proposals --dry-run` to test."""
    print(f">>> smoke ssh check first")
    smoke_ssh_to_e2e.remote()
    print(f">>> push (only={only or 'ALL'}, dry_run={dry_run})")
    result = push_caches_to_e2e.remote(only=only, dry_run=dry_run)
    print(f">>> {result}")
