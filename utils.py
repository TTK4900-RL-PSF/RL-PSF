import numpy as np
from pandas import DataFrame
import gym_rl_mpc.utils.model_params as params
import sys

def simulate_episode(env, agent, max_time, verbose=False, id=None):
    """
    Simulate one episode and return the resulting info/states.
    :param env: Environment to simulate in
    :param agent: Agent to use for simulation
    :param max_time: maximum simulation time
    :return: (DataFrame)
    """
    state_labels = [r"theta", r"theta_dot", r"omega"]
    input_labels = [r"F_thr", r"blade_pitch", r"power"]
    agent_action_labels = [r"agent_F_thr", r"agent_blade_pitch", r"agent_power"]
    psf_action_labels = [r"psf_F_thr", r"psf_blade_pitch", r"psf_power"]
    wind_labels = [r"F_w", r"Q_w", r"Q_gen", r"wind_speed", r"adjusted_wind_speed"]
    reward_labels = [r"reward", r"theta_reward", r"theta_dot_reward", r"omega_reward", r"omega_dot_reward", r"power_reward", r"psf_reward"]
    
    labels = np.hstack(["time", state_labels, input_labels, agent_action_labels, psf_action_labels, wind_labels, reward_labels])

    done = False
    env.reset()
    init_blade_pitch = env.turbine.input[1]/params.max_blade_pitch
    while not done and env.t_step < max_time/env.step_size:
        if agent is not None:
            action, _states = agent.predict(env.observation, deterministic=True)
        else:
            if env.t_step == 0:
                thr = 0
                blade_pitch = init_blade_pitch
                power = params.power_regime(env.wind_speed)
            elif env.t_step in range(0,500):
                thr = 0
                blade_pitch = params.max_blade_pitch
                power = params.power_regime(env.wind_speed)
            else:
                thr = 0
                blade_pitch = init_blade_pitch
                power = params.power_regime(env.wind_speed)
                
            action = np.array([thr, blade_pitch, power])
        _, _, done, _ = env.step(action)
        if verbose:
            report_msg = '{:<20}{:<20}{:<20.2f}{:<20.2%}\r'.format(id, env.t_step, env.cumulative_reward, env.t_step*env.step_size/max_time)
            sys.stdout.write(report_msg)
            sys.stdout.flush()
    time = np.array(env.episode_history['time']).reshape((env.t_step, 1))
    last_reward = np.array(env.episode_history['last_reward']).reshape((env.t_step, 1))
    theta_reward = np.array(env.episode_history['theta_reward']).reshape((env.t_step, 1))
    theta_dot_reward = np.array(env.episode_history['theta_dot_reward']).reshape((env.t_step, 1))
    omega_reward = np.array(env.episode_history['omega_reward']).reshape((env.t_step, 1))
    omega_dot_reward = np.array(env.episode_history['omega_dot_reward']).reshape((env.t_step, 1))
    power_reward = np.array(env.episode_history['power_reward']).reshape((env.t_step, 1))
    psf_reward = np.array(env.episode_history['psf_reward']).reshape((env.t_step, 1))
    states = env.episode_history['states']
    input = env.episode_history['input']
    agent_actions = env.episode_history['agent_actions']
    psf_actions = env.episode_history['psf_actions']
    wind_force = np.array(env.episode_history['wind_force']).reshape((env.t_step, 1))
    wind_torque = np.array(env.episode_history['wind_torque']).reshape((env.t_step, 1))
    generator_torque = np.array(env.episode_history['generator_torque']).reshape((env.t_step, 1))
    wind_speed = np.array(env.episode_history['wind_speed']).reshape((env.t_step, 1))
    adjusted_wind_speed = np.array(env.episode_history['adjusted_wind_speed']).reshape((env.t_step, 1))

    sim_data = np.hstack([  time,
                            states,
                            input,
                            agent_actions,
                            psf_actions,
                            wind_force,
                            wind_torque,
                            generator_torque,
                            wind_speed,
                            adjusted_wind_speed,
                            last_reward,
                            theta_reward,
                            theta_dot_reward,
                            omega_reward,
                            omega_dot_reward,
                            power_reward,
                            psf_reward
                        ])
    df = DataFrame(sim_data, columns=labels)
    return df