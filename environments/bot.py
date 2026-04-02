import random
from minigrid.utils.baby_ai_bot import BabyAIBot

class BabyAIBotEpsilonGreedy:
    def __init__(self, env, random_action_prob = 0., num_actions = None):
        self.expert = BabyAIBot(env)
        self.random_action_prob = random_action_prob

        if num_actions is not None:
            self.num_actions = num_actions
        else:
            self.num_actions = env.action_space.n
        self.last_action = None

    def sample(self, prob):
        return random.random() < prob

    def __call__(self, state):
        if self.sample(self.random_action_prob):
            action = random.randint(0, self.num_actions, ())
        else:
            action = self.expert.replan(self.last_action)

        self.last_action = action
        return action