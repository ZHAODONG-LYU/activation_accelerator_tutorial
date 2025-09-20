# gen\_bf16\_vectors — README (EN)

A utility script to **generate bfloat16 (bf16) test vectors** covering seven ops: **Softmax, LayerNorm, RMSNorm, SiLU, GELU, Elementwise Add, Elementwise Multiply**. It produces fixed-shape tensors (default **N=64, D=768**), exports **bf16 bitstreams (uint16)** for FPGA ingestion, and (optionally) **PyTorch reference outputs** in both bf16 and float32.

---

## Features

* Build **X\_test\_tensor** and **Y\_test\_tensor** (Y is used for Add/Mul on rows 46–53).
* Save **bf16 tensors** and their **uint16 bit patterns** (`.bin`/`.npy`) for AXI MM/DDR.
* (Optional) Emit **reference outputs** per operator (bf16 & f32).
* Generate a **manifest.json** describing the scenario of each row.

---

## Requirements

* Python ≥ 3.8
* PyTorch (1.10+ / 2.x recommended)
* NumPy

> Some PyTorch versions (e.g., 2.4.1) **lack `torch.nanmax`**. The script includes **compat shims** (`nanmax`/`nanmean`) so it works without upgrading.

---

## Quick Start

```bash
# Only generate X/Y bf16 vectors and bit patterns (no references)
python gen_bf16_vectors.py --outdir ./bf16_vectors

# Generate X/Y + reference outputs for all seven ops (bf16 & f32)
python gen_bf16_vectors.py --outdir ./bf16_vectors --emit_ref

# Reproducible randomness
python gen_bf16_vectors.py --outdir ./bf16_vectors --emit_ref --seed 42
```

**Common args**

* `--outdir`  Output directory (required)
* `--emit_ref` Emit PyTorch references
* `--seed`    RNG seed (default 1234)
* `--N`, `--D` Tensor shape (default `N=64`, `D=768`)

---

## Outputs

Example tree (`--outdir ./bf16_vectors`):

```
bf16_vectors/
  ├─ X_test_tensor_bf16.bin         # X as bf16 bitstream (uint16, little-endian, row-major)
  ├─ Y_test_tensor_bf16.bin         # Y as bf16 bitstream (uint16, little-endian, row-major)
  ├─ X_test_tensor_bf16_bits.npy    # X bit patterns [N,D] (uint16)
  ├─ Y_test_tensor_bf16_bits.npy    # Y bit patterns [N,D] (uint16)
  ├─ X_test_tensor_bf16.pt          # X tensor (torch.bfloat16) [N,D]
  ├─ Y_test_tensor_bf16.pt          # Y tensor (torch.bfloat16) [N,D]
  ├─ manifest.json                  # Row → scenario description
  ├─ ref_softmax_bf16.pt            # ↓ appear only with --emit_ref
  ├─ ref_softmax_f32.pt
  ├─ ref_layernorm_bf16.pt
  ├─ ref_layernorm_f32.pt
  ├─ ref_rmsnorm_bf16.pt
  ├─ ref_rmsnorm_f32.pt
  ├─ ref_silu_bf16.pt
  ├─ ref_silu_f32.pt
  ├─ ref_gelu_bf16.pt
  ├─ ref_gelu_f32.pt
  ├─ ref_add_bf16.pt
  └─ ref_mul_bf16.pt
     ref_add_f32.pt
     ref_mul_f32.pt
```

* **Bitstream format**: `uint16`, **little-endian**, row-major; each element is a bf16 1-8-7 layout.
* **Size**: `N × D × 2` bytes. Default `64×768×2 = 98,304` bytes (nicely 128-bit aligned).

---

## Row Mapping (high level)

* **0–11**: Basics & special values (all zeros, ±1, ±BF16\_MAX, ±BF16\_MIN\_NORMAL, mixed specials, alternating ±0).
* **12–19**: Random distributions (various ranges, injected Inf/NaN/subnormals, large positives).
* **20–25**: Softmax stress (large increasing, large decreasing, one dominant, Inf/NaN cases).
* **26–31**: LayerNorm focus (zero-mean, tiny variance, large/small magnitudes, with NaN).
* **32–37**: RMSNorm focus (analogous cases).
* **38–41**: SiLU focus.
* **42–45**: GELU focus.
* **46–53**: **Elementwise Add/Mul X** (script auto-builds the paired **Y** for overflow/underflow/Inf+(-Inf)/0\*Inf/NaN tests).
* **54–63**: Extra / stress (log-normal, extreme alternations, alt seeds).

See `manifest.json` for exact per-row notes.

---


## Inspecting `.bin`

**Shell (hex view, 16 words/line):**

If needed, you can use our provided `test.py` script to inspect the generated vectors. Make sure to replace the `path` with your actual file location.


---

## License

Free to use for research and engineering validation. Attribution appreciated.


