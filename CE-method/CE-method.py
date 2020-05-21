import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import numpy as np

EP_BATCH = 16
HIDDEN_SIZE = 128

Episode = namedtuple('Episode', field_names=['reward', 'observation', 'action'])


class CEAgent(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(CEAgent, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.seq(x)

    def act(self, obs):
        act_scores = self.__call__(obs)
        actions_prob = torch.softmax(act_scores, dim=1).data.numpy()[0]
        action = np.random.choice(len(actions_prob), p=actions_prob)
        return action


def get_batch(env, agent, n_episodes=16, selection_func=None):
    """
        Play n_episodes, select the best episodes then return a tuple (torch.Tensor, torch.Tensor, float), the
        first tensor on containing all the observations from the selected episodes, the second containing all the actions
        taken and the float being the mean reward on all episodes played. If selection_func is None the top 30% are
        selected.
    :param env: The environment used to get the observations.
    :param agent: The agent the interact with the environment.
    :param n_episodes: Number of episodes played by the agent.
    :param selection_func: Function used to select the episodes.
    :return: (torch.Tensor, torch.Tensor, float).
    """

    batch = []
    for i in range(n_episodes):
        observations = []
        actions = []
        obs = env.reset()
        obs_t = torch.Tensor([obs])
        total_reward = 0.0
        is_done = False
        while not is_done:
            action = agent.act(obs_t)
            observations.append(obs)
            actions.append(action)
            obs, reward, is_done, _ = env.step(action)
            obs_t = torch.Tensor([obs])
            total_reward += reward
        batch.append(Episode(reward=total_reward, observation=observations, action=actions))

    reward_mean = np.mean([episode.reward for episode in batch])
    if selection_func:
        selected = selection_func(batch)
    else:
        min_reward = np.percentile([episode.reward for episode in batch], 70)

        selected = [episode for episode in batch if episode.reward >= min_reward]

    selected_observations = [obs for episode in selected for obs in episode.observation]
    selected_actions = [action for episode in selected for action in episode.action]

    observations_tensor = torch.FloatTensor(selected_observations)
    actions_tensor = torch.LongTensor(selected_actions)

    return observations_tensor, actions_tensor, reward_mean


def early_stop():
    return False


def train_loop(env, agent, criterion, n_epochs=100, ep_batch=16, writer=None):
    epoch = 1
    loss = 0
    reward_mean = -500
    optimizer = agent.optimizer
    while epoch < n_epochs and reward_mean < -80:
        batch_x, batch_y, reward_mean = get_batch(env, agent, n_episodes=ep_batch)
        optimizer.zero_grad()
        actions_scores = agent(batch_x)
        loss = criterion(actions_scores, batch_y)
        loss.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f" % (epoch, loss.item(), reward_mean))
        if writer:
            writer.add_scalar("loss", loss.item(), epoch)
            writer.add_scalar("reward_mean", reward_mean, epoch)
        epoch += 1


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = CEAgent(obs_size, HIDDEN_SIZE, n_actions)
    agent.optimizer = torch.optim.Adam(params=agent.parameters(), lr=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(comment="-acrobot")
    train_loop(env, agent, criterion, writer=writer)
    writer.close()
    print("trained")
