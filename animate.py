import argparse
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

import gym_rl_mpc
from gym_rl_mpc.utils import model_params as params
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM


def animate(frame):
    plt.cla()
    height = params.L
    if frame == 0:
        thr = 0
        blade_pitch = init_blade_pitch
        power = params.power_regime(env.wind_speed)
    else:
        thr = 0
        blade_pitch = init_blade_pitch  # + 0.1*DEG2RAD/params.max_blade_pitch
        power = params.power_regime(env.wind_speed)

    if args.data:
        # If reading from file:
        x_top = height * np.sin(data_angle[frame])
        y_top = -(height * np.cos(data_angle[frame]))
        action = data_input[0][frame] / params.max_thrust_force
    else:
        if args.agent:
            action, _states = agent.predict(env.observation, deterministic=True)
        else:
            action = np.array([thr, blade_pitch, power])

        _, _, done, _ = env.step(action)

        x_top = height * np.sin(env.turbine.platform_angle)
        y_top = height * np.cos(env.turbine.platform_angle)
        x_bottom = -params.L_thr * np.sin(env.turbine.platform_angle)
        y_bottom = -params.L_thr * np.cos(env.turbine.platform_angle)
        recorded_states.append(env.turbine.state)
        recorded_inputs.append(env.turbine.input)
        recorded_disturbance.append(np.array(
            [env.turbine.wind_force, env.turbine.wind_torque, env.turbine.generator_torque,
             env.turbine.adjusted_wind_speed]))

    x = [x_bottom, x_top]
    y = [y_bottom, y_top]
    ax_ani.set(xlim=(-0.7 * height, 0.7 * height), ylim=(-1.1 * params.L_thr, 1.2 * height))
    ax_ani.set_xlabel('$X$')
    ax_ani.set_ylabel('$Y$')

    # Plot water line
    ax_ani.plot([-0.6 * height, 0.6 * height], [0, 0], linewidth=1, linestyle='--')

    # Plot pole
    ax_ani.plot(x, y, color='b', linewidth=2)
    # Plot arrow proportional to input force
    ax_ani.arrow(x=-params.L_thr * np.sin(env.turbine.platform_angle),
                 y=-params.L_thr * np.cos(env.turbine.platform_angle),
                 dx=100 * env.turbine.input[0] / params.max_thrust_force, dy=0, head_width=2, head_length=2,
                 length_includes_head=True)
    # Plot arrow proportional to wind force
    ax_ani.arrow(x=x_top, y=y_top, dx=30 * (env.turbine.wind_force / params.max_wind_force), dy=0, head_width=2,
                 head_length=2, length_includes_head=True)
    # Plot wind arrow with wind number
    ax_ani.arrow(x=-50, y=params.L, dx=20, dy=0, head_width=2, head_length=2, length_includes_head=True)
    ax_ani.arrow(x=-50, y=params.L - 10, dx=20, dy=0, head_width=2, head_length=2, length_includes_head=True)
    ax_ani.text(-49, params.L - 7, f"{env.wind_speed:.1f} m/s", fontsize=10)
    # Plot rotational speed
    ax_ani.text(0.35 * height, 1.11 * height, f"$\Omega$ = {env.turbine.omega * (60 / (2 * np.pi)):.2f} rpm",
                fontsize=10)
    # Plot blad pitch bar
    bar_length_multiplier = 2
    min_blade_pitch = -params.min_blade_pitch_ratio * params.max_blade_pitch * RAD2DEG * bar_length_multiplier
    max_blade_pitch = params.max_blade_pitch * RAD2DEG * bar_length_multiplier
    blade_pitch_input = env.turbine.blade_pitch * RAD2DEG * bar_length_multiplier
    ax_ani.broken_barh([(min_blade_pitch, max_blade_pitch), (0, blade_pitch_input)], [1.1 * height, 9],
                       facecolors=('blue', 'red'))
    ax_ani.text(-0.5 * height, 1.11 * height, f"Blade pitch = {env.turbine.blade_pitch * RAD2DEG:.2f} deg", fontsize=10)
    # Plot power bar
    bar_length_multiplier = 2e-6
    max_power = params.max_power_generation * bar_length_multiplier
    power_input = env.turbine.input[2] * bar_length_multiplier
    ax_ani.broken_barh([(30, max_power), (30, power_input)], [0.5 * height, 9], facecolors=('blue', 'red'))
    ax_ani.text(0.2 * height, 0.57 * height, f"Power = {env.turbine.input[2] / 1e6:.2f} MW", fontsize=10)


if __name__ == "__main__":
    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot(111)

    recorded_states = []
    recorded_inputs = []
    recorded_disturbance = []

    parser = argparse.ArgumentParser()
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument(
        '--data',
        help='Path to data .csv file.',
    )
    parser_group.add_argument(
        '--agent',
        help='Path to agent .zip file.',
    )
    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=gym_rl_mpc.SCENARIOS.keys(),
        help="Environment to run."
    )
    parser.add_argument(
        '--time',
        type=int,
        default=50,
        help='Max simulation time (seconds).',
    )
    parser.add_argument(
        '--save_video',
        help='Save animation as mp4 file',
        action='store_true'
    )
    parser.add_argument(
        '--psf',
        help='Use psf corrected action',
        action='store_true'
    )
    args = parser.parse_args()

    if args.data:
        # If file specified, read data from file and animate
        data = pd.read_csv(args.data)
        data_angle = data['theta']
        data_input = np.array([data['F']])
        data_reward = np.array(data['reward'])
        env_id = args.env
    else:
        done = False
        if args.psf:
            config = gym_rl_mpc.SCENARIOS[args.env]['config'].copy()
            config['use_psf'] = True
        else:
            config = gym_rl_mpc.SCENARIOS[args.env]['config']
        if config['use_psf']:
            print("Using PSF corrected actions")
        env = gym.make(args.env, env_config=config)
        env_id = env.unwrapped.spec.id
        env.reset()
        if args.agent:
            agent = PPO.load(args.agent)
        init_blade_pitch = env.turbine.input[1] / params.max_blade_pitch

    print("Animating...")
    animation_speed = 10
    ani = FuncAnimation(fig_ani, animate, interval=1000 * env.step_size / animation_speed, blit=False)

    plt.tight_layout()
    if args.save_video:
        agent_path_list = args.agent.split("\\")
        video_dir = os.path.join("logs", env_id, agent_path_list[-3], "videos")
        os.makedirs(video_dir, exist_ok=True)
        i = 0
        video_path = os.path.join(video_dir, agent_path_list[-1][0:-4] + f"_animation_{i}.mp4")
        while os.path.exists(video_path):
            i += 1
            video_path = os.path.join(video_dir, agent_path_list[-1][0:-4] + f"_animation_{i}.mp4")
        ani.save(video_path, dpi=150)
    plt.show()

    if not args.data:
        recorded_states = np.array(recorded_states)
        recorded_inputs = np.array(recorded_inputs)
        recorded_disturbance = np.array(recorded_disturbance)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        if args.env == 'ConstantWind-v17':
            fig.suptitle(f"Wind speed: {env.wind_speed:.1f} m/s")
        else:
            fig.suptitle(f"Wind mean: {env.wind_mean:.1f} m/s, Wind amplitude: {env.wind_amplitude:.1f} m/s")

        time = np.array(range(0, len(recorded_states[:, 0]))) * env.step_size

        ax1.plot(time, recorded_states[:, 0] * RAD2DEG, label='$\\theta$')
        ax1.plot(time, recorded_states[:, 1] * RAD2DEG, label='$\dot{\\theta}$')
        ax1.plot(time, np.zeros(len(time)), linestyle='--', color='k')
        ax1.set_ylabel('Degrees, deg/sec')
        ax1.set_title('platform angle and angular velocity')
        ax1.legend()

        ax2.plot(time, recorded_states[:, 2] * RAD2RPM, label='$\Omega$')
        ax2.set_ylabel('rpm')
        ax2.set_title('Angluar Velocity Rotor')
        ax2.legend()

        color = 'tab:blue'
        ax3.plot(time, recorded_inputs[:, 0], label='F_thr', color=color)
        ax3.set_ylabel('F_thr [N]', color=color)
        ax3.set_title('Input')
        ax3.legend()
        ax3.set_ylim([-params.max_thrust_force*1.05,params.max_thrust_force*1.05])

        color = 'tab:orange'
        ax3_2 = ax3.twinx()
        ax3_2.plot(time, recorded_inputs[:, 1] * RAD2DEG, label='Blade pitch', color=color)
        ax3_2.set_ylabel('Blade pitch [Degrees]', color=color)
        ax3_2.legend()
        ax3_2.set_ylim([-params.min_blade_pitch_ratio*params.max_blade_pitch * RAD2DEG*1.05,params.max_blade_pitch * RAD2DEG*1.05])

        color = 'tab:blue'
        ax4.plot(time, recorded_disturbance[:, 0], label='F_w', color=color)
        ax4.plot(time, np.zeros(len(time)), linestyle='--', color='k', linewidth=0.5)
        ax4.set_ylabel('F_w [N]', color=color)
        ax4.set_title('Wind Force and torque')
        ax4.legend()

        color = 'tab:orange'
        ax4_2 = ax4.twinx()
        ax4_2.plot(time, recorded_disturbance[:, 1], label='Q_w', color=color)
        ax4_2.plot(time, recorded_disturbance[:, 2], label='Q_gen', color='tab:red')
        ax4_2.set_ylabel('Q [Nm]', color=color)
        ax4_2.legend()

        ax5.plot(time, recorded_inputs[:, 2], label="Power")
        ax5.set_ylabel('P_gen [W]')
        ax5.set_title('Power generated')
        ax5.legend()
        ax5.set_ylim([-0.1e6,params.max_power_generation*1.05])

        fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
        ax21.plot(time, np.array(env.episode_history['agent_actions'])[:, 0] * params.max_thrust_force,
                  label='agent thrust')
        ax21.plot(time, np.array(env.episode_history['psf_actions'])[:, 0] * params.max_thrust_force,
                  label='PSF thrust')
        ax21.set_ylabel('F_thr [N]')
        ax21.set_title('Commanded Thrust Force')
        ax21.legend()
        ax21.set_ylim([-params.max_thrust_force*1.05,params.max_thrust_force*1.05])

        ax22.plot(time, np.array(env.episode_history['agent_actions'])[:, 2] * params.max_power_generation,
                  label='agent power')
        ax22.plot(time, np.array(env.episode_history['psf_actions'])[:, 2] * params.max_power_generation,
                  label='PSF power')
        ax22.set_ylabel('Power')
        ax22.set_title('Commanded Power')
        ax22.legend()
        ax22.set_ylim([-0.1e6,params.max_power_generation*1.05])

        ax23.plot(time, np.array(env.episode_history['agent_actions'])[:, 1] * params.max_blade_pitch * RAD2DEG,
                  label='agent blade pitch')
        ax23.plot(time, np.array(env.episode_history['psf_actions'])[:, 1] * params.max_blade_pitch * RAD2DEG,
                  label='PSF blade pitch')
        ax23.set_ylabel('Blade pitch [deg]')
        ax23.set_title('Commanded blade pitch')
        ax23.legend()
        ax23.set_ylim([-params.min_blade_pitch_ratio*params.max_blade_pitch * RAD2DEG*1.05,params.max_blade_pitch * RAD2DEG*1.05])

        ax24.plot(time, np.array(env.episode_history['wind_speed']), label='Wind speed')
        ax24.set_ylabel('Wind speed [m/s]')
        ax24.set_title('Wind')
        ax24.legend()
        ax24.set_ylim(4.5,25.5)

        plt.show()
