"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.integrate import solve_ivp
import numpy as np

class CartEnv(gym.Env):
    """
    Description:
        Simple 2D cart motion along frictionless track. The cart starts at (0, 0) and wants to stay in bounds.
    Observation:
        Type: Box(4) --> Box(7) ? ( this part I did not know how to edit correctly )
        Num     Observation               Min                     Max
        0       Cart Position a             -10                     10
        1       Cart Velocity a            -1                      1
        2       Cart Velocity b            -10                      10
        3       Cart Velocity a            -1                      1
    Actions:
        Type: Discrete(3)
        Num   Action
        0     V a > V b
        1     V a < V b
        2     V a = V b
        Note: The amount the velocity that is reduced or increased is not
        fixed
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Any cart Position is more than +/- 10 (sides of the cart reach the edge of
        the display).
        Episode length is greater than 200. <-- did not set up yet
        Solved Requirements: <-- did not set up yet
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.length = 0.15  # distance between point a and b
        self.force_mag_a = 1  # increase speed
        self.force_mag_b = 1  # increase speed
        self.tau = 0.02  # seconds between state updates
        self.thetaCurr = 0 # initial and current angle
        self.kinematics_integrator = 'euler'

        # Threshold
        self.x_threshold = 5
        self.y_threshold = 5

        # Observation (unsure if this is right)
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.y_threshold * 2,
                         np.finfo(np.float32).max, 0, 0, 0],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        va, vb, theta, xa, xb, ya, yb = self.state
        if action == 0:
            va = 3*self.force_mag_a
            vb = self.force_mag_b
        elif action == 1:
            va = self.force_mag_a
            vb = 3*self.force_mag_b
        elif action == 2:
            va = self.force_mag_a
            vb = self.force_mag_b

        def sol_angle(t, y, vb, va):
            return (vb - va) / self.length

        t_span = [0, self.tau]
        sol = solve_ivp(sol_angle, t_span, [self.thetaCurr], args=(vb, va))

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        if self.kinematics_integrator == 'euler':
            xa = xa + va * costheta * self.tau
            ya = ya + va * sintheta * self.tau
            xb = xb + vb * costheta * self.tau
            yb = yb + vb * sintheta * self.tau
            theta = sol.y[0][-1]
        else:  # semi-implicit euler
            xa = xa + va * costheta * self.tau
            ya = ya + va * sintheta * self.tau
            xb = xb + vb * costheta * self.tau
            yb = yb + vb * sintheta * self.tau
            theta = sol.y[0][-1]

        self.thetaCurr = theta
        self.state = (va, vb, theta, xa, xb, ya, yb)

        done = bool(
            xa < -self.x_threshold
            or xa > self.x_threshold
            or xb < -self.x_threshold
            or xb > self.x_threshold
            or ya < -self.y_threshold
            or ya > self.y_threshold
            or yb < -self.y_threshold
            or yb > self.y_threshold
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Cart just exited!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(7,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        cartwidth = 20.0
        cartheight = 10.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

        if self.state is None:
            return None

        x = self.state
        cartx = ((x[3] + x[4]) * scale + screen_width) / 2.0  # MIDDLE OF CART
        carty = ((x[5] + x[6]) * scale + screen_height) / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None