import gym
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, Counter


class VISAgent:
    def __init__(self):
        self.rewards = defaultdict(float)
        self.transits = defaultdict(Counter)
        self.values = defaultdict(float)

    def explore(self, env, n_steps):
        state = env.get_current_state()
        for i in range(n_steps):
            action = env.action_space.sample()
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            state = env.reset() if is_done else new_state

    def action_value(self, state, action, gamma=0.9):
        action_value = 0.0
        next_state_counts = self.transits[(state, action)]
        total_transitions = sum(next_state_counts.values())
        for next_state, count in next_state_counts.items():
            transition_prob = count / total_transitions
            reward = self.rewards[(state, action, next_state)]
            long_reward = self.values[next_state]
            action_value += transition_prob * (reward + gamma * long_reward)
        return action_value

    def value_iteration(self, env):
        for state in range(env.observation_space.n):
            actions_values = []
            for action in range(env.action_space.n):
                action_value = self.action_value(state, action)
                actions_values.append(action_value)
            self.values[state] = max(actions_values)

    def act(self, obs, env):
        action = max([(action, self.action_value(obs, action)) for action in range(env.action_space.n)],
                     key=lambda x: x[1])[0]
        return action

    def test(self, env, n_episodes=20):
        total_reward = 0.0
        for i in range(n_episodes):
            state = env.reset()
            is_done = False
            while not is_done:
                action = self.act(state, env)
                new_state, reward, is_done, _ = env.step(action)
                self.rewards[(state, action, new_state)] = reward
                self.transits[(state, action)][new_state] += 1
                state = new_state
                total_reward += reward
        return total_reward/n_episodes


class CurrentStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_state = None

    def observation(self, obs):
        self.current_state = obs
        return obs

    def get_current_state(self):
        return self.current_state


def train_loop():
    writer = SummaryWriter(comment="-v-iteration")

    env = gym.make("FrozenLake-v0")
    env = CurrentStateWrapper(env)
    env.reset()

    agent = VISAgent()

    reward_mean = 0
    epoch = 1
    while reward_mean < 0.80:
        # Step 2
        agent.explore(env, 100)
        # Step 3
        agent.value_iteration(env)
        # Step 4
        test_env = gym.make("FrozenLake-v0")
        reward_mean = agent.test(test_env)
        writer.add_scalar("reward_mean", reward_mean, epoch)
        print("%d: Reward: %.3f" % (epoch, reward_mean))
        epoch += 1
    print("trained")


if __name__ == "__main__":
    train_loop()
