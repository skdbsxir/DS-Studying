import numpy as np

class ReplayBufer:
    def __init__(self):
        self.mem_count = 0

        self.states = np.zeros()