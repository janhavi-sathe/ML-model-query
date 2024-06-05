import env_loader
from pettingzoo import ParallelEnv


def run_env(env: ParallelEnv, dict_agents, num_episodes, max_steps, seed):
  total_returns = {a_name: [] for a_name in dict_agents}
  for _ in range(num_episodes):
    jo_obs, infos = env.reset(seed)
    jo_p_lat = {aname: dict_agents[aname].PREV_LATENT for aname in env.agents}
    dummy_aux = {aname: dict_agents[aname].PREV_AUX for aname in env.agents}
    episode_rewards = {aname: 0 for aname in env.agents}
    done = False

    while not done:
      jo_lat = {}
      jo_action = {}
      for a_name in env.agents:
        agent = dict_agents[a_name]
        latent, action = agent.choose_action(jo_obs[a_name],
                                             jo_p_lat[a_name],
                                             prev_aux=dummy_aux[a_name],
                                             sample=False)
        jo_lat[a_name] = latent
        jo_action[a_name] = action

      jo_obs, rewards, dones, truncates, infos = env.step(jo_action)
      for a_name in env.agents:
        episode_rewards[a_name] += rewards[a_name]
        if dones[a_name] or truncates[a_name]:
          done = True

      jo_p_lat = jo_lat

    for a_name in env.agents:
      total_returns[a_name].append(episode_rewards[a_name])

  return total_returns


if __name__ == "__main__":
  pass
