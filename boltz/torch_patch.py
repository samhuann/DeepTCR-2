import torch
torch.set_float32_matmul_precision('medium')
print("✅ Patched: torch float32 matmul precision set to", torch.get_float32_matmul_precision())
