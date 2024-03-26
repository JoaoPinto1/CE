import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

map_4_by_4 = generate_random_map(size=4, seed=42)
map_8_by_8 = generate_random_map(size=8, seed=42)
map_12_by_12 = generate_random_map(size=12, seed=42)


MAX_ITERATIONS_4_by_4 = 100
MAX_ITERATIONS_8_by_8 = 200
MAX_ITERATIONS_12_by_12 = 500