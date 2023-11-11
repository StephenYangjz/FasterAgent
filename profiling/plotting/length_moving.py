# plot

# data
# 1. 256 tokens, 11.8, 11.6
# 2. 512 tokens, 21.3, 20.4
# 3. 1024 tokens, 39.4, 37.7
# 4. 2048 tokens, 75.8, 72.1
# 5. 2560 tokens, 273.8, 90.5
# 6. 3072 tokens, 325.1, 107.2
# 7. 4096 tokens, 590.4, 143.1

from matplotlib import pyplot as plt
import numpy as np

x = np.array([256, 512, 1024, 2048, 2560, 3072, 4096])
y = np.array([11.8, 21.3, 39.4, 75.8, 273.8, 325.1, 590.4])
y2 = np.array([11.6, 20.4, 37.7, 72.1, 90.5, 107.2, 143.1])

plt.plot(x, y, label="GPU to CPU")
plt.plot(x, y2, label="CPU to GPU")
plt.xlabel("Max Sequence Length")
plt.ylabel("Time (ms)")
plt.title("Time to Copy KV Cache")
plt.legend()
plt.savefig("kv_cache_copy.png")