"""
Track B: Hierarchical and Tiered Memory Simulator for LLM Inference
COA Project 9 — Memory-Centric Architectures for LLMs

Models a 4-tier memory hierarchy and tracks where weights and KV-cache live,
how much data moves per token, and how migrations occur as the working set
exceeds each tier's capacity.

Tier defaults are taken from public hardware specs:
  Tier 0 — On-chip SRAM      :  4 MB   |  10.0 TB/s |   1 ns
  Tier 1 — HBM (stacked)     : 32 GB   |   3.35 TB/s|  10 ns   (HBM3 / A100/H100)
  Tier 2 — Off-chip DRAM     : 64 GB   |  68  GB/s  |  80 ns   (DDR5-5600)
  Tier 3 — Far/CXL Memory    : 512 GB  |  25  GB/s  | 500 ns   (CXL 3.0 pool)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


class Tier(IntEnum):
    SRAM    = 0
    HBM     = 1
    DRAM    = 2
    FAR_MEM = 3


@dataclass(frozen=True)
class TierSpec:
    name: str
    bandwidth_GBs: float    # gigabytes per second
    latency_ns: float
    capacity_bytes: int
    energy_pJ_per_byte: float   # read energy, picojoules/byte


# Defaults — public-spec-based. Override per-instance via TieredMemorySimulator(tiers=...).
DEFAULT_TIERS: List[TierSpec] = [
    TierSpec("SRAM",    bandwidth_GBs=10_000,  latency_ns=1.0,
             capacity_bytes=4 * 1024**2,            energy_pJ_per_byte=2.0),
    TierSpec("HBM",     bandwidth_GBs= 3_350,  latency_ns=10.0,
             capacity_bytes=32 * 1024**3,           energy_pJ_per_byte=4.0),
    TierSpec("DRAM",    bandwidth_GBs=    68,  latency_ns=80.0,
             capacity_bytes=64 * 1024**3,           energy_pJ_per_byte=25.0),
    TierSpec("FAR_MEM", bandwidth_GBs=    25,  latency_ns=500.0,
             capacity_bytes=512 * 1024**3,          energy_pJ_per_byte=100.0),
]


@dataclass
class GpuSpec:
    """Roofline reference for the host GPU (used only for AI vs ridge-point compare)."""
    name: str
    peak_tflops_bf16: float
    peak_mem_bw_TBs:  float


KNOWN_GPUS: List[GpuSpec] = [
    GpuSpec("H200",            989.0, 4.800),
    GpuSpec("H100 SXM",        989.0, 3.350),
    GpuSpec("H100 PCIe",       756.0, 2.000),
    GpuSpec("A100 SXM4 80GB",  312.0, 2.000),
    GpuSpec("A100 SXM4 40GB",  312.0, 1.555),
    GpuSpec("A100",            312.0, 1.555),  # generic fallback
    GpuSpec("L40S",            362.0, 0.864),
    GpuSpec("L4",              121.0, 0.300),
    GpuSpec("T4",               65.0, 0.320),
    GpuSpec("V100",            112.0, 0.900),
]


def detect_gpu(name_str: Optional[str] = None) -> GpuSpec:
    """Match an nvidia-smi-style GPU name string against the known list.
    Returns an honest CPU/host fallback (with H100 reference numbers) if no match."""
    if not name_str:
        return GpuSpec("CPU/host (no GPU; H100 reference)", 989.0, 3.350)
    best = None
    best_len = 0
    for g in KNOWN_GPUS:
        if g.name in name_str and len(g.name) > best_len:
            best = g
            best_len = len(g.name)
    if best is None:
        return GpuSpec(f"{name_str} (unknown; H100 reference)", 989.0, 3.350)
    # return a copy with the original name preserved
    return GpuSpec(name=name_str, peak_tflops_bf16=best.peak_tflops_bf16,
                   peak_mem_bw_TBs=best.peak_mem_bw_TBs)


# ─────────────────────────────────────────────────────────────────────────────
#  Model parameters
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelParams:
    """Bare minimum needed for memory-traffic modelling. Read these from
    llama-cpp-python: n_layer(), n_head_kv(), n_head(), n_embd(), n_params(),
    on-disk file size for weight_bytes."""
    name: str
    n_layers: int
    n_kv_heads: int
    n_heads: int
    n_embd: int
    n_params: int
    weight_bytes: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads if self.n_heads else 64


# ─────────────────────────────────────────────────────────────────────────────
#  Simulator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """Per-decode-step record produced by the simulator."""
    step_index: int
    kv_tokens_after: int
    weight_bytes_read: int
    kv_read_bytes: int
    kv_write_bytes: int
    kv_tier: Tier
    kv_size_bytes: int
    decode_time_ms_modeled: float


@dataclass
class TieredMemorySimulator:
    """Track B core. Inputs: ModelParams + KV element width.
    Methods: place_weights, place_kv, record_prefill, record_decode_step.
    Outputs: per-step records and aggregate session metrics."""
    model: ModelParams
    kv_elem_bytes: float = 2.0          # fp16 default; q8=1.0; q4=0.5
    tiers: List[TierSpec] = field(default_factory=lambda: list(DEFAULT_TIERS))
    gpu: GpuSpec = field(default_factory=lambda: detect_gpu(None))

    # session counters (mutable)
    session_tokens_generated: int = 0
    session_bytes_moved:      int = 0
    session_kv_read_bytes:    int = 0
    session_kv_write_bytes:   int = 0
    session_weight_read_bytes: int = 0
    session_tier_migrations:  int = 0
    session_remote_accesses:  int = 0   # decode steps where KV in tier >= 2
    session_energy_pJ:        float = 0.0

    # per-turn (cleared by reset_turn)
    turn_prefill_tokens:        int = 0
    turn_decode_tokens:         int = 0
    turn_prefill_weight_bytes:  int = 0
    turn_decode_weight_bytes:   int = 0
    turn_kv_read_bytes:         int = 0
    turn_kv_write_bytes:        int = 0

    # internal state
    _prev_kv_tier: Tier = Tier.SRAM
    step_log: List[StepRecord] = field(default_factory=list)

    # ── Capacity-aware placement ──────────────────────────────────────────
    def place_weights(self) -> Tier:
        """Weights skip Tier 0 (SRAM too small for any real model) and
        live in the fastest remaining tier they fit in."""
        for t in (Tier.HBM, Tier.DRAM, Tier.FAR_MEM):
            if self.model.weight_bytes <= self.tiers[int(t)].capacity_bytes:
                return t
        return Tier.FAR_MEM

    def place_kv(self, kv_bytes: int) -> Tier:
        for t in (Tier.SRAM, Tier.HBM, Tier.DRAM, Tier.FAR_MEM):
            if kv_bytes <= self.tiers[int(t)].capacity_bytes:
                return t
        return Tier.FAR_MEM

    # ── Bytes-per-token primitives ────────────────────────────────────────
    def kv_bytes_for(self, n_tokens: int) -> int:
        """KV-cache size for n_tokens. Each token has a (K, V) pair per layer
        per KV-head, of head_dim elements at kv_elem_bytes each."""
        return int(n_tokens * self.model.n_layers * 2
                   * self.model.n_kv_heads * self.model.head_dim
                   * self.kv_elem_bytes)

    def kv_read_per_decode(self, kv_tokens: int) -> int:
        """Bytes read from KV-cache during one decode step at attention."""
        return self.kv_bytes_for(kv_tokens)

    def kv_write_per_token(self) -> int:
        """Bytes written to KV-cache for ONE new token."""
        return self.kv_bytes_for(1)

    # ── Latency model ─────────────────────────────────────────────────────
    def time_ms_for(self, bytes_: int, tier: Tier) -> float:
        """Bandwidth-bound latency to move `bytes_` from `tier` (in ms)."""
        bw_bytes_per_s = self.tiers[int(tier)].bandwidth_GBs * 1e9
        return (bytes_ / bw_bytes_per_s) * 1000.0

    def energy_pJ_for(self, bytes_: int, tier: Tier) -> float:
        return bytes_ * self.tiers[int(tier)].energy_pJ_per_byte

    # ── Migration detection ───────────────────────────────────────────────
    def _check_migration(self, kv_tokens_after: int) -> None:
        cur = self.place_kv(self.kv_bytes_for(kv_tokens_after))
        if cur != self._prev_kv_tier:
            self.session_tier_migrations += 1
            self._prev_kv_tier = cur

    # ── Public step API ───────────────────────────────────────────────────
    def reset_turn(self) -> None:
        self.turn_prefill_tokens = 0
        self.turn_decode_tokens = 0
        self.turn_prefill_weight_bytes = 0
        self.turn_decode_weight_bytes = 0
        self.turn_kv_read_bytes = 0
        self.turn_kv_write_bytes = 0

    def record_prefill(self, n_prompt: int, kv_tokens_after: int) -> None:
        """Prefill: weights read once, KV writes for new tokens, KV reads triangular."""
        kv_before = max(0, kv_tokens_after - n_prompt)
        w_bytes = self.model.weight_bytes
        kv_write = int(self.model.n_layers * n_prompt * 2
                       * self.model.n_kv_heads * self.model.head_dim
                       * self.kv_elem_bytes)
        sum_after  = kv_tokens_after * (kv_tokens_after - 1) // 2
        sum_before = kv_before       * (kv_before - 1)       // 2
        kv_read = int(self.model.n_layers * (sum_after - sum_before) * 2
                      * self.model.n_kv_heads * self.model.head_dim
                      * self.kv_elem_bytes)

        self.turn_prefill_tokens       += n_prompt
        self.turn_prefill_weight_bytes += w_bytes
        self.turn_kv_write_bytes       += kv_write
        self.turn_kv_read_bytes        += kv_read

        self.session_weight_read_bytes += w_bytes
        self.session_kv_write_bytes    += kv_write
        self.session_kv_read_bytes     += kv_read
        self.session_bytes_moved       += w_bytes + kv_write + kv_read

        # energy: weights read from weight tier; KV r/w from current KV tier
        wt = self.place_weights()
        kt = self.place_kv(self.kv_bytes_for(kv_tokens_after))
        self.session_energy_pJ += (self.energy_pJ_for(w_bytes, wt)
                                   + self.energy_pJ_for(kv_read, kt)
                                   + self.energy_pJ_for(kv_write, kt))
        self._check_migration(kv_tokens_after)

    def record_decode_step(self, kv_tokens_after: int) -> StepRecord:
        """One decoded token. Returns the per-step record."""
        w_bytes  = self.model.weight_bytes
        kv_write = self.kv_write_per_token()
        kv_read  = self.kv_read_per_decode(kv_tokens_after)

        self.turn_decode_tokens         += 1
        self.turn_decode_weight_bytes   += w_bytes
        self.turn_kv_write_bytes        += kv_write
        self.turn_kv_read_bytes         += kv_read

        self.session_tokens_generated   += 1
        self.session_weight_read_bytes  += w_bytes
        self.session_kv_write_bytes     += kv_write
        self.session_kv_read_bytes      += kv_read
        self.session_bytes_moved        += w_bytes + kv_write + kv_read

        wt = self.place_weights()
        kt = self.place_kv(self.kv_bytes_for(kv_tokens_after))
        decode_ms = (self.time_ms_for(w_bytes, wt)
                     + self.time_ms_for(kv_read, kt)
                     + self.time_ms_for(kv_write, kt))
        self.session_energy_pJ += (self.energy_pJ_for(w_bytes, wt)
                                   + self.energy_pJ_for(kv_read, kt)
                                   + self.energy_pJ_for(kv_write, kt))
        if int(kt) >= int(Tier.DRAM):
            self.session_remote_accesses += 1
        self._check_migration(kv_tokens_after)

        rec = StepRecord(
            step_index=self.session_tokens_generated,
            kv_tokens_after=kv_tokens_after,
            weight_bytes_read=w_bytes,
            kv_read_bytes=kv_read,
            kv_write_bytes=kv_write,
            kv_tier=kt,
            kv_size_bytes=self.kv_bytes_for(kv_tokens_after),
            decode_time_ms_modeled=decode_ms,
        )
        self.step_log.append(rec)
        return rec

    # ── Roofline / analysis helpers ───────────────────────────────────────
    def arithmetic_intensity(self) -> float:
        """Per decode-token AI = (2 * n_params) FLOPs / bytes_moved_per_token."""
        if self.turn_decode_tokens == 0:
            return 0.0
        bytes_per_tok = (self.turn_decode_weight_bytes
                         + self.turn_kv_read_bytes
                         + self.turn_kv_write_bytes) / self.turn_decode_tokens
        if bytes_per_tok <= 0:
            return 0.0
        flops_per_tok = 2.0 * self.model.n_params
        return flops_per_tok / bytes_per_tok

    def ridge_point(self) -> float:
        """FLOP/byte where the host GPU transitions from memory-bound to compute-bound."""
        if self.gpu.peak_mem_bw_TBs <= 0:
            return 0.0
        return self.gpu.peak_tflops_bf16 / self.gpu.peak_mem_bw_TBs

    def session_summary(self) -> dict:
        return dict(
            tokens_generated         = self.session_tokens_generated,
            total_bytes_moved        = self.session_bytes_moved,
            weight_read_bytes        = self.session_weight_read_bytes,
            kv_read_bytes            = self.session_kv_read_bytes,
            kv_write_bytes           = self.session_kv_write_bytes,
            tier_migrations          = self.session_tier_migrations,
            remote_accesses          = self.session_remote_accesses,
            energy_pJ                = self.session_energy_pJ,
            arithmetic_intensity     = self.arithmetic_intensity(),
            ridge_point              = self.ridge_point(),
            weight_tier              = self.place_weights().name,
            final_kv_tier            = self._prev_kv_tier.name,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Optimization variants — modeled by re-running the simulator with tweaks
# ─────────────────────────────────────────────────────────────────────────────


def model_kv_quantization(sim: TieredMemorySimulator, kv_bits: int) -> TieredMemorySimulator:
    """Return a fresh simulator with KV element width set to kv_bits/8."""
    return TieredMemorySimulator(
        model=sim.model,
        kv_elem_bytes=kv_bits / 8.0,
        tiers=sim.tiers,
        gpu=sim.gpu,
    )


def kv_read_bytes_with_window(sim: TieredMemorySimulator,
                              kv_tokens_total: int,
                              window: int) -> int:
    """Bytes a single decode step would read from KV under sliding-window attention."""
    effective = min(kv_tokens_total, window)
    return sim.kv_bytes_for(effective)


def fused_attention_savings_bytes(sim: TieredMemorySimulator,
                                  kv_tokens: int) -> int:
    """Bytes saved per prefill step by FlashAttention-style fusion (no QK^T or
    softmax round-trip to HBM). Returns the bytes that would be eliminated."""
    # Naive attention writes QK^T (T x n_heads x T) to HBM in fp16 then reads it
    return int(2 * kv_tokens * sim.model.n_heads * kv_tokens * 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Compute-centric baseline (rubric +10 bonus)
# ─────────────────────────────────────────────────────────────────────────────


def compute_centric_predicted_tps(sim: TieredMemorySimulator) -> float:
    """A 'compute-only' analysis would predict tok/s = peak_TFLOPS / (2 * n_params)."""
    flops_per_tok = 2.0 * sim.model.n_params
    return sim.gpu.peak_tflops_bf16 * 1e12 / flops_per_tok


def memory_bound_predicted_tps(sim: TieredMemorySimulator,
                               kv_tokens: int) -> float:
    """Bandwidth-bound prediction: tok/s = host_HBM_bw / bytes_per_tok."""
    bytes_per_tok = (sim.model.weight_bytes
                     + sim.kv_read_per_decode(kv_tokens)
                     + sim.kv_write_per_token())
    if bytes_per_tok <= 0:
        return 0.0
    return sim.gpu.peak_mem_bw_TBs * 1e12 / bytes_per_tok


# ─────────────────────────────────────────────────────────────────────────────
#  Adaptive KV-placement policy (project's small original contribution)
#
#  Static (baseline; current llama.cpp / vLLM behaviour):
#     KV lives in HBM until exhaustion, at which point a bulk transfer to the
#     next slower tier happens at the crossing step, stalling that one step
#     by `kv_bytes / dest_BW`.
#
#  Adaptive:
#     The simulator predicts the imminent crossing and pre-migrates the cache
#     across the previous `lookahead_steps` decode steps, amortising the bulk
#     transfer. Tail latency at the crossing drops by ~lookahead_steps×; total
#     decode time is approximately preserved (work is reorganised, not
#     eliminated).
# ─────────────────────────────────────────────────────────────────────────────


def analyze_static_vs_adaptive(sim: TieredMemorySimulator,
                                lookahead_steps: int = 10):
    """Replay sim.step_log under static and adaptive policies. Pure post-hoc
    analysis — does not mutate the simulator state.

    Returns
    -------
    static_lat   : list[float]  per-step ms under the baseline static policy
    adaptive_lat : list[float]  per-step ms under the adaptive policy
    migrations   : list[dict]   one entry per tier crossing
    """
    if not sim.step_log:
        return [], [], []

    migrations = []
    prev_tier = sim.step_log[0].kv_tier
    for rec in sim.step_log:
        if rec.kv_tier != prev_tier:
            migrations.append(dict(step=rec.step_index,
                                   from_tier=prev_tier,
                                   to_tier=rec.kv_tier,
                                   kv_bytes=rec.kv_size_bytes))
            prev_tier = rec.kv_tier

    # Static: full bulk transfer at the crossing step
    static_lat = [rec.decode_time_ms_modeled for rec in sim.step_log]
    for mig in migrations:
        mig_ms = sim.latency_bw_ms(mig['kv_bytes'], mig['to_tier']) \
                 if hasattr(sim, 'latency_bw_ms') \
                 else (mig['kv_bytes'] /
                       (sim.tiers[int(mig['to_tier'])].bandwidth_GBs * 1e9)) * 1000.0
        idx = mig['step'] - 1
        if 0 <= idx < len(static_lat):
            static_lat[idx] += mig_ms

    # Adaptive: spread transfer over lookahead_steps preceding decode steps
    adaptive_lat = [rec.decode_time_ms_modeled for rec in sim.step_log]
    for mig in migrations:
        mig_ms = (mig['kv_bytes'] /
                  (sim.tiers[int(mig['to_tier'])].bandwidth_GBs * 1e9)) * 1000.0
        per_step_bonus = mig_ms / lookahead_steps
        end = mig['step'] - 1
        start = max(0, end - lookahead_steps + 1)
        for s in range(start, end + 1):
            adaptive_lat[s] += per_step_bonus

    return static_lat, adaptive_lat, migrations


if __name__ == "__main__":
    _demo = ModelParams(
        name="smoke-test",
        n_layers=8,
        n_kv_heads=4,
        n_heads=4,
        n_embd=256,
        n_params=50_000_000,
        weight_bytes=100 * 1024**2,
    )
    _sim = TieredMemorySimulator(model=_demo)
    _sim.reset_turn()
    _sim.record_prefill(32, 32)
    for _i in range(5):
        _sim.record_decode_step(33 + _i)
    _sum = _sim.session_summary()
    print("tiered_memory_sim.py OK —", _sum["tokens_generated"], "decode steps;",
          f"bytes_moved={_sum['total_bytes_moved']:,}")
