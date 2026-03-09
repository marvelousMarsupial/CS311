import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
bins = [
    np.linspace(-4.8, 4.8, 10),
    np.linspace(-4, 4, 10),
    np.linspace(-0.42, 0.42, 10),
    np.linspace(-4, 4, 10),
]
Q = np.zeros([10, 10, 10, 10, 2])


def discretize(state):
    return tuple(np.digitize(state[i], bins[i]) - 1 for i in range(4))


rewards = []
for ep in range(500):
    s, _ = env.reset()
    eps = max(0.01, 1 - ep / 300)
    total_r = 0
    for _ in range(500):
        a = (
            env.action_space.sample()
            if np.random.rand() < eps
            else Q[discretize(s)].argmax()
        )
        s2, r, term, trunc, _ = env.step(a)
        Q[discretize(s)][a] += 0.1 * (
            r + 0.99 * Q[discretize(s2)].max() - Q[discretize(s)][a]
        )
        total_r += r
        s = s2
        if term or trunc:
            break
    rewards.append(total_r)
    if ep % 50 == 0:
        print(
            f"Episode {ep:4d} | Avg Reward (last 50): {np.mean(rewards[-50:]):6.1f} | Eps: {eps:.3f}"
        )
