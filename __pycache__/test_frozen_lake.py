import gymnasium as gym
import random
from maps_to_evaluate import *
RENDER_MODE = 'human'
env = gym.make('FrozenLake-v1', desc=map_4_by_4, is_slippery=False, render_mode=RENDER_MODE)
observation, info = env.reset(seed=42)
if RENDER_MODE is not None:
    env.render()
for step in range(MAX_ITERATIONS_4_by_4):
    action = random.randint(0,3)
    observation, reward, terminated, truncated, info = env.step(action)
    if RENDER_MODE is not None:
        env.render()
    if terminated:
        break