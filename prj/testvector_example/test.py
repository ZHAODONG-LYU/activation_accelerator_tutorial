import numpy as np, torch

N, D = 64, 768
path = "/data1/jcz/FPGA/bf16_vectors/X_test_tensor_bf16.bin"

# 读作 uint16 位型并 reshape
bits = np.fromfile(path, dtype=np.uint16).reshape(N, D)

# 还原 bfloat16 -> float32 （仅用于数值查看）
t_bf16 = torch.from_numpy(bits).view(torch.bfloat16)
t_f32  = t_bf16.to(torch.float32).numpy()

def view_rows(rows, hex_count=16, float_count=8):
    """一次查看多行：打印前 hex_count 个 16-bit 位型（HEX）和前 float_count 个数值"""
    for r in rows:
        if r < 0 or r >= N:
            print(f"[skip] row {r} 超出范围 [0,{N-1}]")
            continue
        hex_words = " ".join(f"{w:04X}" for w in bits[r, :hex_count])
        floats    = ", ".join(f"{v:.6g}" for v in t_f32[r, :float_count])
        print(f"\n== Row {r} ==")
        print(f"HEX (first {hex_count}):   {hex_words}")
        print(f"F32 (first {float_count}): {floats}")

# 例：一次查看 8 行（0~7）
view_rows(rows=range(0, 64), hex_count=16, float_count=16)