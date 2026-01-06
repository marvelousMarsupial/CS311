import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size=64):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)
