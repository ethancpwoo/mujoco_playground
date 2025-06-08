import jax
import mediapy as media

from jax import numpy as jp
from mujoco_playground import registry

env = registry.load('ManipulatorBringBall')
env_cfg = registry.get_default_config('ManipulatorBringBall')

# env = registry.load('CartpoleBalance')
# env_cfg = registry.get_default_config('CartpoleBalance')

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))
rollout = [state]
print("got here")

f = 0.5
for i in range(env_cfg.episode_length):
  action = []
  for j in range(env.action_size):
    action.append(
        jp.sin(
            state.data.time * 2 * jp.pi * f + j * 2 * jp.pi / env.action_size
        )
    )
  action = jp.array(action)
  print("looped before step")
  state = jit_step(state, action)
  rollout.append(state)
  print("looped finished")

print("got here 2")

frames = env.render(rollout)
media.write_video("manip_video.mp4", frames, fps=1.0 / env.dt)