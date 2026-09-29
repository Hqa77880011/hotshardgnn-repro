"""Small local CPU launcher for platforms whose torchrun lacks libuv support."""

import argparse
import os
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("module")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.nproc < 1:
        parser.error("nproc must be positive")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = []
    try:
        for rank in range(args.nproc):
            env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(args.nproc),
                       MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0", OMP_NUM_THREADS="1")
            processes.append(subprocess.Popen([sys.executable, "-m", args.module, *args.args], env=env))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise SystemExit("A worker failed; stopping the remaining workers")
            time.sleep(.1)
        if any(p.returncode for p in processes):
            raise SystemExit(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
