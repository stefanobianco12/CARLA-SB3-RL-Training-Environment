import os
import argparse
import pandas as pd
import numpy as np
import config
import torch
from attack import fgsm_attack,PGD,critic,MAD

parser = argparse.ArgumentParser(description="Eval a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--model", type=str, default="", required=True, help="Path to a model evaluate")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--no_record_video", action="store_false", help="If True, record video of the evaluation")
parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")
parser.add_argument("--atk", type=str, default=None)

args = vars(parser.parse_args())
config.set_config(args["config"])

from stable_baselines3 import PPO, DDPG, SAC

from utils import VideoRecorder, parse_wrapper_class
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions

from vae.utils.misc import LSIZE
from carla_env.wrappers import vector, get_displacement_vector
from carla_env.envs.carla_route_env import CarlaRouteEnv
from eval_plots import plot_eval, summary_eval

from config import CONFIG
from PIL import Image

def save_image(latent,folder,cont):
    tensor = latent.detach().cpu().squeeze(0)  # [3, 80, 160]

    # Convert to numpy and scale to [0, 255]
    np_image = tensor.permute(1, 2, 0).numpy()  # [80, 160, 3]
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)

    # Create and save image
    img = Image.fromarray(np_image)
    img.save(f"./{folder}/{cont}.png")


def run_eval(env, model, vae, model_path=None, record_video=False):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval')
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"
    agent_name=model_id.split('_')[0]
    # vec_env = model.get_env()
    state = env.reset()
    rendered_frame = env.render(mode="rgb_array")

    columns = ["model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
               "reward", "distance", "speed", "center_dev", "angle_next_waypoint", "waypoint_x", "waypoint_y",
               "route_x", "route_y"]
    df = pd.DataFrame(columns=columns)

    # Init video recording
    if record_video:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
                                                              int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                       frame_size=rendered_frame.shape,
                                       fps=env.fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = 0
    # While non-terminal state
    print("Episode ", episode_idx)
    saved_route = False
    cont=0
    while episode_idx < 6:
        env.extra_info.append("Evaluation")
        if args['atk'] is not None:
            state_tensor=model.policy.obs_to_tensor(state)  
            if agent_name== "PPO":
                logit, values, log_prob= model.policy(state_tensor[0], deterministic=True)
                action,_states= model.predict(state, deterministic=True)
                action=torch.tensor(action, dtype=torch.float32).unsqueeze(0).to('cuda')
            elif agent_name=="SAC":
                action, logit, log_std = model.actor.grad_forward_pass(state_tensor[0], deterministic=False)

            
            if args['atk']=="FGSM":
                state=fgsm_attack(model,state_tensor[0],action,logit,0.4,agent_name)
            elif args['atk']=="PGD":
                state=PGD(model,state_tensor[0],action,0.1,agent_name)
            elif args['atk']=="Critic":
               state=critic(model,state_tensor[0],0.1)
            #state=MAD(model,state_tensor[0],0.1)


        action,_states= model.predict(state, deterministic=True)
        state, reward, dones, info = env.step(action)
        cont+=1
        if env.step_count >= 150 and env.current_waypoint_index == 0:
            dones = True

        # Save route at the beginning of the episode
        if not saved_route:
            initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            initial_vehicle_location = vector(env.vehicle.get_location())
            # Save the route to plot them later
            for way in env.route_waypoints:
                route_relative = get_displacement_vector(initial_vehicle_location,
                                                         vector(way[0].transform.location),
                                                         initial_heading)
                new_row = pd.DataFrame([['route', env.episode_idx, route_relative[0], route_relative[1]]],
                                       columns=["model_id", "episode", "route_x", "route_y"])
                df = pd.concat([df, new_row], ignore_index=True)
            saved_route = True

        vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()),
                                                   initial_heading)
        waypoint_relative = get_displacement_vector(initial_vehicle_location,
                                                    vector(env.current_waypoint.transform.location), initial_heading)

        new_row = pd.DataFrame(
            [[model_id, env.episode_idx, env.step_count, env.vehicle.control.throttle, env.vehicle.control.steer,
              vehicle_relative[0], vehicle_relative[1], reward,
              env.distance_traveled,
              env.vehicle.get_speed(), env.distance_from_center,
              np.rad2deg(env.vehicle.get_angle(env.current_waypoint)),
              waypoint_relative[0], waypoint_relative[1], None, None
              ]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

        # Add frame
        rendered_frame = env.render(mode="rgb_array")
        if record_video:
            video_recorder.add_frame(rendered_frame)
        if dones:
            state = env.reset()
            episode_idx += 1
            saved_route = False
            print("Episode ", episode_idx)

    # Release video
    if record_video:
        video_recorder.release()

    df.to_csv(csv_path, index=False)
    plot_eval([csv_path])
    summary_eval(csv_path)


if __name__ == "__main__":
    model_path = args["model"]

    algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}
    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]

    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

    env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], viewer_res=(1120, 560), host=args["host"], port=args["port"],
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                        fps=args["fps"], action_smoothing=CONFIG["action_smoothing"],
                        action_space_type='continuous', activate_spectator=True, eval=True,
                        activate_render=args["no_render"])

    for wrapper_class_str in CONFIG["wrappers"]:
        wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
        env = wrap_class(env, *wrap_params)

    model = AlgorithmRL.load(model_path, env=env, device='cuda')

    run_eval(env, model, vae, model_path, record_video=args['no_record_video'])
