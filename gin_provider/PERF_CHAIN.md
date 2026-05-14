# DeepEP PP Performance Chain v1 → v5 (final)

Branch `feat/gin-provider-proxy-mode`. All numbers are 3-run medians of
`/tmp/run_deepep_pp_perf_fan.sh 4096 7168 4 1` (4096-token / 7168-hidden,
conc=3, hide=1, 56 MB tensor, NCCL_GIN_FANOUT_MIN=1 MiB), reported as
`recv GB/s` per rank from `tests/elastic/test_pp.py`.

## Summary table

| Step | Commit | Binary fanout | rank0 recv | rank1 recv | Δ vs prev | Δ vs v1 | Status |
|------|--------|---------------|------------|------------|-----------|---------|--------|
| v1   | `ea661c2` | 1 | 33.56 | 34.51 | — | 1.000× / 1.000× | ✅ measured |
| v2   | `10fa0c9` | 1 | 33.40 | 34.51 | -0.5% / 0% | 0.995× / 1.000× | ✅ measured (noise-flat) |
| v2.1 | `b76f0a9` (branch) | 1 | 34.97 | 35.42 | +4.7% / +2.6% | 1.042× / 1.026× | ✅ measured (3-run, σ≈0.5) |
| v3   | `b4aec08` | 4 | 44.32 | 49.26 | +27% / +39% | 1.321× / 1.428× | ✅ measured |
| v4   | `2a01e58` | 3 | 54.45 | 54.12 | +23% / +10% | 1.622× / 1.568× | ✅ measured |
| v5   | `2f8af88` | 3 (target) | (untestable this session) | — | (env-blocked) | — | ⚠ design-only |

Default-fanout=1 regression check on v5: rank0=34.5 / rank1=35.3 GB/s
(3-run median) — within noise of v1/v2.1 ⇒ v5 introduces no regression
on the default path.

## Strict-increase chain (verified)

`v1 → v2.1 → v3 → v4` is empirically strictly increasing on both ranks
at the same workload, ≥3 runs each, same orchestrator, same env vars.

`v4 → v5`: not validated this session. The dxs/TCPXO control-plane
listener on both hosts only accepts 1 of N expected fan-out connections
within the 120 s handshake deadline. Restarting `rxdm` and
`fastrakTestContainer` on both hosts did not recover. **The same v4
binary that produced 54.4/54.1 GB/s in `deepep_perf_v4.md` also fails
this handshake now**, so the blocker is external to v5 — likely a
listener-state regression in dxs after many rapid connect/disconnect
cycles. Track in `gin_provider/README.md` follow-ups.

## v5 design rationale (why we expect ≥1.05× over v4 at fanout≥3)

Six call sites of `static std::atomic<int> dbg{0}; if (dbg.fetch_add(1) < N)`
on hot paths (`proxy_progress.cc::RunInbound` x2, `plugin_main.cc::IputCommon` x4)
were rewritten to `thread_local int dbg_tls = 0`. Even after the log
budget saturates, the atomic RMW still triggers a cross-core cache-line
bounce on every Iput / inbound recv. Under fanout=N with N inbound
threads + multiple NCCL proxy threads the contention shows up directly
as inbound-thread CPU saturation (the ceiling that already capped
fanout=4 at 47 GB/s vs fanout=3 at 54 GB/s in v4).

Empirical validation requires the dxs handshake env to recover. Test
recipe (unchanged from v4):

```bash
NCCL_GIN_FANOUT=3 NCCL_GIN_FANOUT_MIN=1048576 \
  TAG=v5_fan3 /tmp/run_deepep_pp_perf_fan.sh 4096 7168 4 1
```

Acceptance criterion: r0 ≥ 57 GB/s (≥1.05× v4 r0 54.4) over 3-run median.

## Branch hygiene

- `feat/gin-provider-proxy-mode` (HEAD `2f8af88`): v1 .. v5 linear chain.
- `v2.1` (HEAD `b76f0a9`): historical reference for the v2 → v2.1 hdr-wait
  cherry-pick. **Kept**, not merged into main (its content is already in
  v4's `2a01e58` for the main-branch perf line).

## Cross-references

- `deepep_perf_baseline_v1.md`, `deepep_perf_v2.md`, `deepep_perf_v3.md`,
  `deepep_perf_v4.md`, `deepep_perf_v5.md` in `tencent_h100/` repo
  (tracked separately from this fork).
- Implementation log §15 .. §19 — same project repo.
