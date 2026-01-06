import gymnasium as gym
from replay_buffer import ReplayBuffer

# env
env = None
for env_id in ("LunarLander-v2", "LunarLander-v3"):
    try:
        env = gym.make(env_id)
        break
    except Exception:
        pass
if env is None:
    raise RuntimeError("No LunarLander env found (v2/v3).")

buf = ReplayBuffer(50_000)

obs, _ = env.reset()
for _ in range(500):  # collect a few transitions
    a = env.action_space.sample()
    nobs, r, term, trunc, _ = env.step(a)
    done = term or trunc
    buf.push(obs, a, float(r), nobs, done)
    obs = nobs
    if done:
        obs, _ = env.reset()

print("buffer size:", len(buf))
batch = buf.sample(8)
print(
    "sample[0] shapes:", len(batch[0][0]), "->", len(batch[0][3]), "done:", batch[0][4]
)

env.close()
