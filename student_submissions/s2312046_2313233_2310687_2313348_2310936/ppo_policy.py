'''
Description: RL agent trained under PPO for 2D cutting-stock
author: Lương Minh Thuận  - 2313348
Time: Dec, 2024
'''

from .PPOmodule import *
import numpy as np
import matplotlib.pyplot as plt
from policy import Policy


#from torch.utils.tensorboard import SummaryWriter


class PPO(Policy):
    def __init__(self, env=None, load_check_pontis=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 5
        self.gae_lambda = 0.95
        self.rollout_steps = 50  # N - big batch
        self.batch_size = 10
        self.leaning_rate = 0.0003

        self.customObs = []
        self.obsInfo = dict()
        self.filter_out = 1
        self.actor = ActorCritic(self.filter_out, self.leaning_rate)
        # self.critic = CriticNetwork(self.filter_out,self.leaning_rate)
        self.env = env
        if load_check_pontis or env is None:
            self.actor.load_checkpoint()
            # self.critic.load_checkpoint()
        self.memory = PPOMemory(self.batch_size)
        #self.old_action = None  # for inference

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        # self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        # self.critic.load_checkpoint()

    def choose_action(self, observation):
        observation = np.array([observation], dtype=float)
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        dist, value = self.actor(state)
        # value = self.critic([observation])
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def get_action(self, obs, info):
        if len(self.customObs) == 0:
            self.resetCustomObservation(obs)
            #self.stepInfo()
        else:
            self.updateCustomObservation(obs)
        with torch.no_grad():
            #print(self.customObs)
            state = np.array([self.customObs], dtype=float)
            state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
            dist, value = self.actor(state)
            action = torch.argmax(dist.probs).item()
        convertedAction = self.convertAction(obs, action)
        reward, done = self.getCustomReward(obs, action)
        #self.old_action = action
        if done:
            self.customObs = []
        return convertedAction

    def stepInfo(self, postInfo=False):
        count = quantity = 0
        for p in self.obsInfo["productInfo"]:
            count += 1
            quantity += p[3]
        if postInfo:
            print(f'Remaining products: {quantity}, intactStock: {len(self.obsInfo["intact"])}')
        else:
            print(f'---------------------------------------',
                  f'\nTotal product types: {count}, products demand: {quantity}, ')
            # f'total product area: {self.obsInfo["productArea"]}, total product area: {self.obsInfo["stockArea"]}')

    def train(self, total_timestep):
        figure_file = 'rewards.png'
        best_score = -10000
        score_history = []
        learn_iters = 0
        avg_score = 0
        n_steps = 0
        save_freq = total_timestep // 10
        cur_save_step = save_freq
        while n_steps < total_timestep:
            observation, _ = self.env.reset()
            self.resetCustomObservation(observation)
            #
            done = False
            score = 0
            game_timestep = 0
            self.stepInfo()
            while not done:
                action, prob, val = self.choose_action(self.customObs)
                # scaledAction = self.getScaledAction(action)
                convertedAction = self.convertAction(observation, action)
                reward, _ = self.getCustomReward(observation, action)
                observation_, _, done, _, info = self.env.step(convertedAction)

                n_steps += 1
                game_timestep += 1
                score += reward
                self.remember(self.customObs, action, prob, val, reward, done)
                self.updateCustomObservation(observation_)
                if n_steps % self.rollout_steps == 0:
                    self.learn()
                    learn_iters += 1
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            normalized_reward = score / game_timestep
            if normalized_reward > best_score:
                best_score = normalized_reward
                self.save_models()
            if n_steps > cur_save_step:
                cur_save_step += save_freq
                self.save_models()
            self.stepInfo(True)
            print('Game step', game_timestep, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                  'time_steps', n_steps, 'learning_steps', learn_iters)
        x = [i + 1 for i in range(len(score_history))]
        self.save_models()
        self.plot_learning_curve(x, score_history, figure_file)

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist, critic_value = self.actor(states)
                #critic_value = self.critic(state_arr[batch])
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                # self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                # self.critic.optimizer.step()
        self.memory.clear_memory()

    @staticmethod
    def plot_learning_curve(x, scores, figure_file):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(figure_file)

    @staticmethod
    def can_place(stock, stock_size, prod_size, ori_size):
        stock_w, stock_h = stock_size
        prod_w, prod_h = prod_size
        res = False
        mask = np.zeros((stock_w, stock_h), dtype=int)
        bias_mask = np.zeros(stock_size, dtype=float)
        bias_mask.fill(-stock_w - stock_h)
        if prod_h == 0 or prod_w == 0:
            return False, mask, bias_mask
        w, h = ori_size
        for x in range(w):
            for y in range(h):
                if (x + prod_w <= w) and (y + prod_h <= h) and (np.all(
                        stock[x: x + prod_w, y: y + prod_h] == -1)):
                    mask[x, y] = 1
                    bias_mask[x, y] = (stock_w + stock_h - x - y)
                    res = True
                elif (x + prod_h <= w) and (y + prod_w <= h) and (np.all(
                        stock[x: x + prod_h, y: y + prod_w] == -1)):
                    mask[x, y] = 1
                    bias_mask[x, y] = (stock_w + stock_h - x - y)
                    res = True
        return res, mask, bias_mask

    def initCustomObs(self, obs):
        self.customObs = []
        next_product = []  # idx, w, h, quantity
        for product in self.obsInfo["productInfo"]:
            if product[3] > 0:
                next_product.append([product[0], product[1], product[2], product[3]])
                break
        while len(next_product) < 1:
            next_product.append([0, 0, 0, 0])
        chosen_stock = []  # 1 stocks - [i,w,h]
        output_mask = None
        bias_mask = None
        product_size = [next_product[0][1], next_product[0][2]]
        w, h = obs["stocks"][0].shape
        for stock in self.obsInfo["stockInfo"]:
            if stock[3] < product_size[0] * product_size[1]:
                continue
            can_place, mask, bias = self.can_place(obs["stocks"][stock[0]], (w, h), product_size,(stock[1],stock[2]))
            if can_place:
                chosen_stock.append([stock[0], w, h])
                output_mask = mask
                bias_mask = bias
                break
        while len(chosen_stock) < 1:
            chosen_stock.append([-1, -1, -1])
        for stock in chosen_stock:
            if stock[0] > -1:
                s = obs["stocks"][stock[0]][:stock[1], :stock[2]].copy()
                s[s != -1] = 0
                s[s == -1] = 1
                self.customObs.append(s)
            else:
                s = np.full(shape=(w, h), fill_value=0, dtype=int)
                self.customObs.append(s)
        if output_mask is None:
            output_mask = np.full(shape=(w, h), fill_value=0, dtype=int)
        if bias_mask is None:
            bias_mask = np.full(shape=(w, h), fill_value=0, dtype=int)
        chosen_product = next_product[0]
        self.customObs.append(bias_mask)
        self.customObs.append(output_mask)
        self.obsInfo["chosenStock"] = chosen_stock[0]
        self.obsInfo["chosenProduct"] = chosen_product
        #print(chosen_stock)

    def resetCustomObservation(self, obs):
        stocksInfo = []  # idx, w, h, fill_area
        #stockArea = productArea = 0
        for i, stock in enumerate(obs["stocks"]):
            w = np.count_nonzero(stock[:, 0] > -2)
            h = np.count_nonzero(stock[0, :] > -2)
            stocksInfo.append([i, w, h, w * h])
        productInfo = []  # idx, w, h, quantity
        for i, product in enumerate(obs["products"]):
            w, h, quantity = int(product["size"][0]), int(product["size"][1]), product["quantity"]
            productInfo.append([i, w, h, quantity])
        sorted_stock = stocksInfo
        #sorted_product = productInfo
        sorted_product = sorted(productInfo, key=lambda x: x[1] * x[2], reverse=True)
        intactStock = [stock[0] for stock in sorted_stock]
        self.obsInfo = {"stockInfo": sorted_stock, "productInfo": sorted_product, "intact": intactStock,
                        "chosenStock": [], "chosenProduct": []}
        self.initCustomObs(obs)

    def updateCustomObservation(self, newObs):
        # action = np.array(action.squeeze(dim=0).cpu())
        productIdx = self.obsInfo["chosenProduct"][0]
        stockIdx = self.obsInfo["chosenStock"][0]
        for i, product in enumerate(self.obsInfo["productInfo"]):
            if product[0] == productIdx:
                self.obsInfo["productInfo"][i][3] -= 1
                break
        for i, stock in enumerate(self.obsInfo["stockInfo"]):
            if stock[0] == stockIdx:
                w, h = newObs["products"][productIdx]["size"]
                self.obsInfo["stockInfo"][i][3] -= w * h
                break
        if stockIdx in self.obsInfo["intact"]:
            self.obsInfo["intact"].remove(stockIdx)
        self.initCustomObs(newObs)

    def convertAction(self, obs, action):
        stockIdx = self.obsInfo["chosenStock"][0]
        stock = obs["stocks"][stockIdx]
        w, h = self.obsInfo["chosenStock"][1], self.obsInfo["chosenStock"][2]
        x, y = action // h, action % h
        width, height = self.obsInfo["chosenProduct"][1], self.obsInfo["chosenProduct"][2]
        if not (np.all(stock[x: x + width, y: y + height] == -1)) or x + width > w or y + height > h:
            width, height = height, width
        return {"stock_idx": stockIdx, "size": np.array([width, height]), "position": (x, y)}

    def getCustomReward(self, obs, action):
        stockIdx = self.obsInfo["chosenStock"][0]
        if stockIdx < 0:
            return -1, False
        stock = obs["stocks"][stockIdx]
        stockWidth, stockHeight = stock.shape
        x, y = action // stockHeight, action % stockHeight
        reward = (stockWidth + stockHeight - x - y)
        if x > 0 or y > 0:
            if x > 0 and y > 0:
                if stock[x - 1, y - 1] == -1:
                    reward = -1 * x - 1 * y
            elif x > 0:
                if stock[x - 1, y] == -1:
                    reward = -1 * x
            elif y > 0:
                if stock[x, y - 1] == -1:
                    reward = -1 * y
        remain = 0
        done = False
        for product in obs["products"]:
            remain += product["quantity"]
        if remain == 1:
            done = True
            #reward += 10
        return reward / 10, done
