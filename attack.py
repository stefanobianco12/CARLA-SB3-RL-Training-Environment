import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.kl import kl_divergence






def fgsm_attack(model,state, action, logit,epsilon,agent_name):
    loss = F.cross_entropy(logit, action)
    model.policy.zero_grad()
    loss.backward(retain_graph=True)
    vae_latent_grad = state['vae_latent'].grad.data
    perturbed_vae = torch.clamp(
        state['vae_latent'] + epsilon * vae_latent_grad.sign(),
        state['vae_latent'] - epsilon,
        state['vae_latent'] + epsilon
    ).detach()

    # Perturb vehicle_measures
    vehicle_measures_grad = state['vehicle_measures'].grad.data
    perturbed_vehicle = torch.clamp(
        state['vehicle_measures'] + epsilon * vehicle_measures_grad.sign(),
        state['vehicle_measures'] - epsilon,
        state['vehicle_measures'] + epsilon
    ).detach()

    if agent_name=="PPO":
        waypoint_measures_grad = state['waypoints'].grad.data
        perturbed_waypoint = torch.clamp(
        state['waypoints'] + epsilon * waypoint_measures_grad.sign(),
        state['waypoints'] - epsilon,
        state['waypoints'] + epsilon
        ).detach()
        perturbed_waypoint=perturbed_waypoint.squeeze(0).detach().cpu().numpy()
    
    perturbed_vae=perturbed_vae.detach().cpu().numpy().astype('float32').flatten()
    perturbed_vehicle=perturbed_vehicle.detach().cpu().numpy().flatten().tolist()
    
    if agent_name=="PPO":
        perturbed_state = {
        'vae_latent': perturbed_vae,
        'vehicle_measures': perturbed_vehicle,
        'waypoints': perturbed_waypoint
        }
    else:
        perturbed_state = {
        'vae_latent': perturbed_vae,
        'vehicle_measures': perturbed_vehicle,
        'maneuver': state['maneuver'].item()
        }
    return perturbed_state






def MAD(model, state, epsilon):
    K = 20
    eta = 3.5 * epsilon / K

    # Clone the original tensors
    vae_orig = state['vae_latent'].detach()
    vehicle_orig = state['vehicle_measures'].detach()

    # Initialize perturbations randomly within the epsilon ball
    vae_adv = vae_orig + (2 * epsilon) * (torch.rand_like(vae_orig) - 0.5)
    vae_adv = torch.clamp(vae_adv, vae_orig - epsilon, vae_orig + epsilon)

    vehicle_adv = vehicle_orig + (2 * epsilon) * (torch.rand_like(vehicle_orig) - 0.5)
    vehicle_adv = torch.clamp(vehicle_adv, vehicle_orig - epsilon, vehicle_orig + epsilon)

    _,logit_orig,log_std_orig=model.actor.grad_forward_pass(state, deterministic=False)
    print(logit_orig)

    for _ in range(K):
        vae_adv.requires_grad_(True)
        vehicle_adv.requires_grad_(True)

        # Prepare state copy
        state_copy = state.copy()
        state_copy['vae_latent'] = vae_adv
        state_copy['vehicle_measures'] = vehicle_adv

        # Forward pass
        _, logit_perturbed, log_std = model.actor.grad_forward_pass(state_copy, deterministic=False)
        print(logit_perturbed)
        loss = kl_divergence(logit_orig, logit_perturbed)
        print(loss)
        input("ALT")
        # Zero gradients and backward
        model.policy.zero_grad()
        loss.backward(retain_graph=True)

        # Gradient-based updates
        vae_adv_grad = vae_adv.grad.data
        vehicle_adv_grad = vehicle_adv.grad.data

        vae_adv = vae_adv + eta * vae_adv_grad.sign()
        vae_adv = torch.clamp(vae_adv, vae_orig - epsilon, vae_orig + epsilon).detach()

        vehicle_adv = vehicle_adv + eta * vehicle_adv_grad.sign()
        vehicle_adv = torch.clamp(vehicle_adv, vehicle_orig - epsilon, vehicle_orig + epsilon).detach()

    # Convert to numpy as in FGSM
    perturbed_vae = vae_adv.detach().cpu().numpy().astype('float32').flatten()
    perturbed_vehicle = vehicle_adv.detach().cpu().numpy().flatten().tolist()

    perturbed_state = {
        'vae_latent': perturbed_vae,
        'vehicle_measures': perturbed_vehicle,
        'maneuver': state['maneuver'].item()
    }

    return perturbed_state




def PGD(model, state, action, epsilon,agent_name):
    K = 20
    eta = 3.5 * epsilon / K

    # Clone the original tensors
    vae_orig = state['vae_latent'].detach()
    vehicle_orig = state['vehicle_measures'].detach()
    if agent_name=="PPO":
        waypoint_orig=state['waypoints'].detach()

    # Initialize perturbations randomly within the epsilon ball
    vae_adv = vae_orig + (2 * epsilon) * (torch.rand_like(vae_orig) - 0.5)
    vae_adv = torch.clamp(vae_adv, vae_orig - epsilon, vae_orig + epsilon)

    vehicle_adv = vehicle_orig + (2 * epsilon) * (torch.rand_like(vehicle_orig) - 0.5)
    vehicle_adv = torch.clamp(vehicle_adv, vehicle_orig - epsilon, vehicle_orig + epsilon)

    if agent_name=="PPO":
        waypoint_adv = waypoint_orig + (2 * epsilon) * (torch.rand_like(waypoint_orig) - 0.5)
        waypoint_adv = torch.clamp(waypoint_adv, waypoint_orig - epsilon, waypoint_orig + epsilon)

    for _ in range(K):
        vae_adv.requires_grad_(True)
        vehicle_adv.requires_grad_(True)
        if agent_name=="PPO":
            waypoint_adv.requires_grad_(True)

        # Prepare state copy
        state_copy = state.copy()
        state_copy['vae_latent'] = vae_adv
        state_copy['vehicle_measures'] = vehicle_adv
        if agent_name=="PPO":
            state_copy['waypoints'] = waypoint_adv

        # Forward pass
        if agent_name== "PPO":
            logit, values, log_prob= model.policy(state_copy, deterministic=True)
            #action,_states= model.predict(state, deterministic=True)
            #action=torch.tensor(action, dtype=torch.float32).unsqueeze(0).to('cuda')
        else:
            _, logit, log_std = model.actor.grad_forward_pass(state_copy, deterministic=False)
        loss = F.cross_entropy(logit, action)
        # Zero gradients and backward
        model.policy.zero_grad()
        loss.backward(retain_graph=True)

        # Gradient-based updates
        vae_adv_grad = vae_adv.grad.data
        vehicle_adv_grad = vehicle_adv.grad.data
        if agent_name== "PPO":
            waypoint_adv_grad = waypoint_adv.grad.data

        vae_adv = vae_adv + eta * vae_adv_grad.sign()
        vae_adv = torch.clamp(vae_adv, vae_orig - epsilon, vae_orig + epsilon).detach()

        vehicle_adv = vehicle_adv + eta * vehicle_adv_grad.sign()
        vehicle_adv = torch.clamp(vehicle_adv, vehicle_orig - epsilon, vehicle_orig + epsilon).detach()

        if agent_name== "PPO":
            waypoint_adv = waypoint_adv + eta * waypoint_adv_grad.sign()
            waypoint_adv = torch.clamp(waypoint_adv, waypoint_orig - epsilon, waypoint_orig + epsilon).detach()

    # Convert to numpy as in FGSM
    perturbed_vae = vae_adv.detach().cpu().numpy().astype('float32').flatten()
    perturbed_vehicle = vehicle_adv.detach().cpu().numpy().flatten().tolist()
    if agent_name== "PPO":
        perturbed_waypoint=waypoint_adv.squeeze(0).detach().cpu().numpy()

    if agent_name=="PPO":
        perturbed_state = {
        'vae_latent': perturbed_vae,
        'vehicle_measures': perturbed_vehicle,
        'waypoints': perturbed_waypoint
        }
    else:
        perturbed_state = {
        'vae_latent': perturbed_vae,
        'vehicle_measures': perturbed_vehicle,
        'maneuver': state['maneuver'].item()
        }

    return perturbed_state

def critic(model,state,epsilon):
    K=50
    eta=epsilon/K
     # Clone the original tensors
    vae_orig = state['vae_latent'].detach()
    vehicle_orig = state['vehicle_measures'].detach()

    # Initialize perturbations randomly within the epsilon ball
    vae_adv = vae_orig + (2 * epsilon) * (torch.rand_like(vae_orig) - 0.5)
    vae_adv = torch.clamp(vae_adv, vae_orig - epsilon, vae_orig + epsilon)

    vehicle_adv = vehicle_orig + (2 * epsilon) * (torch.rand_like(vehicle_orig) - 0.5)
    vehicle_adv = torch.clamp(vehicle_adv, vehicle_orig - epsilon, vehicle_orig + epsilon)

    for _ in range(K):
        vae_adv.requires_grad_(True)
        vehicle_adv.requires_grad_(True)

        # Prepare state copy
        state_copy = state.copy()
        state_copy['vae_latent'] = vae_adv
        state_copy['vehicle_measures'] = vehicle_adv

        action, logit, log_std = model.actor.grad_forward_pass(state_copy, deterministic=False)
        #ris=model.critic()
        q_value=model.critic(state,action)[0]
        model.critic.zero_grad()
        q_value.backward(retain_graph=True)
        vae_adv_grad = vae_adv.grad.data
        vehicle_adv_grad = vehicle_adv.grad.data

        vae_adv = vae_adv + eta * vae_adv_grad.sign()
        vae_adv = torch.clamp(vae_adv, vae_orig - epsilon, vae_orig + epsilon).detach()

        vehicle_adv = vehicle_adv + eta * vehicle_adv_grad.sign()
        vehicle_adv = torch.clamp(vehicle_adv, vehicle_orig - epsilon, vehicle_orig + epsilon).detach()

    # Convert to numpy as in FGSM
    perturbed_vae = vae_adv.detach().cpu().numpy().astype('float32').flatten()
    perturbed_vehicle = vehicle_adv.detach().cpu().numpy().flatten().tolist()

    perturbed_state = {
        'vae_latent': perturbed_vae,
        'vehicle_measures': perturbed_vehicle,
        'maneuver': state['maneuver'].item()
    }

    return perturbed_state



