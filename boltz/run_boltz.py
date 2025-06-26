import torch
torch.set_float32_matmul_precision('medium')  # ← force it manually

import subprocess
import sys

# Call the real boltz CLI with all arguments
subprocess.run(["boltz"] + sys.argv[1:])
