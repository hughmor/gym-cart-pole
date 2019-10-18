import gym
from agent import CartPoleAgent

environment = gym.make('CartPole-v1')
learning_agent = CartPoleAgent(environment)


def train_agent(env, agent, n_episodes=1000):
    for ep in range(n_episodes):
        evaluate_episode(env, agent)
        # update agent
    env.close()
    return agent


def evaluate_episode(env, agent, n_steps=1000):
    observation = env.reset()
    for i in range(n_steps):
        env.render()
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()


trained_agent = train_agent(environment, learning_agent)
