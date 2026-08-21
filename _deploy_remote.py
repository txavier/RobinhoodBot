#!/usr/bin/env python3
"""One-shot deploy of the two patched files to the remote build host, then
build/push/rollout. All deployment targets (host, user, remote dir, password)
are read from the environment so no personal/private info (LAN IP, username,
filesystem paths) is ever committed to the repo. Password is passed inline
over SSH, never written to disk. Mirrors deploy.sh cmd_build + cmd_deploy_bot
minus rsync.

Required env vars:
  RHOST_HOST       Remote build host (e.g. ssh reachable hostname/IP)
  RHOST_USER       SSH username on the remote host
  RHOST_REMOTE_DIR Absolute path to the repo checkout on the remote host
  RHOST_PASS       SSH password for the remote host
"""
import os, sys, io, time
import paramiko

HOST = os.environ.get("RHOST_HOST")
USER = os.environ.get("RHOST_USER")
REMOTE_DIR = os.environ.get("RHOST_REMOTE_DIR")
PASS = os.environ.get("RHOST_PASS")

# Local files are derived from this script's own location so the repo path is
# never hardcoded. Maps local path -> remote relative path inside REMOTE_DIR.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REMOTE_REL_PATHS = [
    "robinhoodbot/main.py",
    "robinhoodbot/robin_stocks/robinhood/authentication.py",
]
LOCAL_FILES = [
    (os.path.join(REPO_ROOT, *rel.split("/")), rel)
    for rel in REMOTE_REL_PATHS
]

_missing = [name for name, val in (
    ("RHOST_HOST", HOST),
    ("RHOST_USER", USER),
    ("RHOST_REMOTE_DIR", REMOTE_DIR),
    ("RHOST_PASS", PASS),
) if not val]
if _missing:
    sys.exit("Missing required env vars: " + ", ".join(_missing))

def ssh_connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, password=PASS, timeout=30,
                look_for_keys=False, allow_agent=False)
    return cli

def sftp_upload(cli):
    sftp = cli.open_sftp()
    for local, remote_rel in LOCAL_FILES:
        remote_path = f"{REMOTE_DIR}/{remote_rel}"
        print(f"[upload] {local} -> {remote_path}")
        # ensure remote dir exists
        d = os.path.dirname(remote_path)
        try:
            sftp.stat(d)
        except IOError:
            sftp.mkdir(d)
        sftp.put(local, remote_path)
    sftp.close()

def run(cli, cmd, timeout=600):
    print(f"\n[run] $ {cmd}")
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    # stream output as it arrives
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
    # drain stderr
    err = stderr.read().decode(errors="replace")
    if err:
        sys.stdout.write(err)
    exit_code = stdout.channel.recv_exit_status()
    print(f"[exit] {exit_code}")
    return exit_code

def main():
    cli = ssh_connect()
    print("[ok] SSH connected")
    sftp_upload(cli)
    print("[ok] files uploaded")

    # Verify remote python compiles the patched files
    run(cli, f"cd {REMOTE_DIR} && python3 -m py_compile robinhoodbot/main.py "
             f"robinhoodbot/robin_stocks/robinhood/authentication.py && echo REMOTE_COMPILE_OK")

    # Build + push image on remote (matches cmd_build)
    build = (f"cd {REMOTE_DIR} && docker build -t robinhoodbot:latest . && "
             f"docker tag robinhoodbot:latest localhost:5000/robinhoodbot:latest && "
             f"docker push localhost:5000/robinhoodbot:latest")
    ec = run(cli, build, timeout=1800)
    if ec != 0:
        sys.exit(f"[fail] docker build/push exited {ec}")

    # Apply manifests + rollout (matches cmd_deploy_bot, config secret already exists)
    ksteps = [
        f"kubectl apply -f {REMOTE_DIR}/k8s/namespace.yaml",
        f"kubectl apply -f {REMOTE_DIR}/k8s/priority-classes.yaml",
        f"kubectl apply -f {REMOTE_DIR}/k8s/node-setup-daemonset.yaml",
        f"kubectl apply -f {REMOTE_DIR}/k8s/robinhoodbot-data-longhorn-pvc.yaml",
        f"kubectl apply -f {REMOTE_DIR}/k8s/deployment.yaml",
        f"kubectl rollout restart deployment/robinhoodbot -n robinhoodbot",
    ]
    for s in ksteps:
        ec = run(cli, s, timeout=180)
        if ec != 0:
            sys.exit(f"[fail] kubectl step exited {ec}")

    # Wait for rollout
    run(cli, f"kubectl rollout status deployment/robinhoodbot -n robinhoodbot --timeout=180s", timeout=200)
    print("\n[done] Deploy issued. Tail logs with: kubectl logs -f deployment/robinhoodbot -n robinhoodbot")
    cli.close()

if __name__ == "__main__":
    main()
