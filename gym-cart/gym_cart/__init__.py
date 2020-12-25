from gym.envs.registration import register

register(
    id='cart-v0',
    entry_point='gym_cart.envs:CartEnv',
)