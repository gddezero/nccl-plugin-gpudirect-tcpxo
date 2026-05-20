"""M1 E2E test — 2 ranks across 2 nodes:
   - init NCCL ProcessGroup
   - create deep_ep.ElasticBuffer (smallest possible)
   - This forces:
       ncclCommQueryProperties  (checks ginType != NONE)
       ncclDevCommCreate        (PROXY mode → our plugin->listen, connect)
       ncclCommWindowRegister   (→ our plugin->regMrSym)
   - Success criterion: NO exception. The ElasticBuffer constructor returns.

Set env before running:
  RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT / LOCAL_RANK
  NCCL_DEBUG=INFO  NCCL_DEBUG_SUBSYS=INIT,NET
  NCCL_NET_PLUGIN=/work/epv2_gin_plugin/build_deploy/libnccl-net-tcpxo-gin.so
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    print(f"[r{rank}] init_process_group world_size={world_size} "
          f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
          flush=True)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"[r{rank}] init_process_group OK", flush=True)

    # Import after dist init so PyTorch NCCL is wired.
    import deep_ep
    print(f"[r{rank}] deep_ep imported (v{deep_ep.__version__})", flush=True)

    # Smallest viable ElasticBuffer.
    try:
        buf = deep_ep.ElasticBuffer(
            dist.group.WORLD,
            num_bytes=4 * 1024 * 1024,  # 4 MiB symmetric window
            explicitly_destroy=True,
        )
        print(f"[r{rank}] ElasticBuffer created OK", flush=True)
        del buf
        print(f"[r{rank}] ElasticBuffer destroyed OK", flush=True)
        ok = True
    except Exception as e:
        print(f"[r{rank}] ElasticBuffer FAILED: {e}", flush=True)
        traceback.print_exc()
        ok = False

    dist.barrier()
    dist.destroy_process_group()
    print(f"[r{rank}] {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
