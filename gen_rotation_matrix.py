import numpy as np
q , r = np.linalg.qr(np.random.randn(32*32*3,32*32*3))
np.save("rotation.npy",q)