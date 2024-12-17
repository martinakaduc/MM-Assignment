from policy import Policy
from policy import GreedyPolicy, RandomPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class Policy2210517_2210050_2210077_2210707(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Initialize the chosen policy
        if policy_id == 1:
            self.policy = ColGenPolicy()
        elif policy_id == 2:
            self.policy = A2CAgent()

    def get_action(self, observation, info):
        """Ensure the policy is reinitialized if needed (in case of environment reset)."""
        # Check if the policy is an instance of ColGenPolicy and reinitialize if necessary
        if isinstance(self.policy, ColGenPolicy):
            # Ensure ColGenPolicy's state is updated
            self.policy.update_state(observation)  # Update the state with the new observation

        # if isinstance(self.policy, A2CAgent):
        #     # Ensure ColGenPolicy's state is updated
        #     self.policy.update_state(observation)  # Update the state with the new observation
        # Get action from the selected policy
        return self.policy.get_action(observation, info)


class ColGenPolicy(Policy):
    def __init__(self):
        super().__init__()
        # Initialize class variables
        self.products = None  # List of products
        self.stock_size = None  # Dimensions of the stock (width, height)
        self.patterns = None  # Patterns for column generation

    def update_state(self, observation):
        """Update the internal state of the policy based on the new observation."""
        # This method will update the attributes based on the new observation
        self.products = observation["products"]
        stock_example = observation["stocks"][0]  # Use the first stock example
        self.stock_size = self._get_stock_size_(stock_example)
        self.patterns = []  # Clear and update patterns if necessary

    def initialize(self, observation):
        """Initialize the policy with observation data (called only once)."""
        self.update_state(observation)  # Initialize using the update_state method

    def get_action(self, observation, info):
        """Determine the next action, ensure the policy state is updated."""
        # Update the state with the latest observation data
        self.update_state(observation)

        # Initial placeholders for the action
        selected_stock_idx, selected_x, selected_y = -1, None, None
        selected_product_size = None
        optimal_placement = None

        # Iterate over all products to determine placement
        for product in self.products:
            if product["quantity"] > 0:  # Only consider products with remaining quantity
                product_size = product["size"]
                min_placement_score = float("inf")  # Track the best score for placement
                current_best_placement = None  # Best placement configuration for this product

                # Iterate over all available stocks
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_width, stock_height = self._get_stock_size_(stock)
                    product_width, product_height = product_size

                    # Check placement in regular orientation
                    pos_x, pos_y = self._find_position_(stock, product_size)
                    if pos_x is not None and pos_y is not None:
                        remaining_width = stock_width - pos_x - product_width
                        remaining_height = stock_height - pos_y - product_height
                        wasted_area = (stock_width * stock_height) - (product_width * product_height)
                        usable_area_ratio = max(remaining_width, 0) * max(remaining_height, 0) / (stock_width * stock_height)
                        placement_score = wasted_area + (1 - usable_area_ratio)  # Scoring based on waste and usability

                        # Update best score and placement for this product
                        if placement_score < min_placement_score:
                            min_placement_score = placement_score
                            current_best_placement = (stock_idx, pos_x, pos_y, product_size)

                    # Check placement in rotated orientation
                    pos_x, pos_y = self._find_position_(stock, product_size[::-1])
                    if pos_x is not None and pos_y is not None:
                        remaining_width = stock_width - pos_x - product_height
                        remaining_height = stock_height - pos_y - product_width
                        wasted_area = (stock_width * stock_height) - (product_height * product_width)
                        usable_area_ratio = max(remaining_width, 0) * max(remaining_height, 0) / (stock_width * stock_height)
                        placement_score = wasted_area + (1 - usable_area_ratio)  # Scoring based on waste and usability

                        # Update best score and placement for rotated product
                        if placement_score < min_placement_score:
                            min_placement_score = placement_score
                            current_best_placement = (stock_idx, pos_x, pos_y, product_size[::-1])

                # If a valid placement is found, update the optimal placement
                if current_best_placement:
                    optimal_placement = current_best_placement
                    break  # Stop searching once the first valid placement is found

        # If a valid placement is found, update the selected placement details
        if optimal_placement:
            selected_stock_idx, selected_x, selected_y, selected_product_size = optimal_placement
        else:
            # If no valid placement was found, handle the case appropriately
            selected_stock_idx, selected_x, selected_y, selected_product_size = -1, 0, 0, (0, 0)  # Default action

        # Return the placement action
        return {
            "stock_idx": selected_stock_idx,
            "size": selected_product_size,
            "position": (selected_x, selected_y),
        }

    def _find_position_(self, stock, product_size):
        """Find a position in the stock where the product can be placed."""
        stock_width, stock_height = self._get_stock_size_(stock)
        product_width, product_height = product_size

        # Iterate over possible positions in the stock
        for x in range(stock_width - product_width + 1):
            for y in range(stock_height - product_height + 1):
                # Check if the product can be placed at this position
                if self._can_place_(stock, (x, y), product_size):
                    return x, y  # Return valid position

        # Return None if no valid position is found
        return None, None


class VectorStateE(nn.Module):
    def __init__(self, total_stock, High_Weight, High_Height, total_product, max_in_one):
        super(VectorStateE, self).__init__()
        self.total_stock = total_stock
        self.High_Weight = High_Weight
        self.High_Height = High_Height
        self.total_product = total_product
        self.max_in_one = max_in_one

        self.cnnN1 = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)  
        self.cnnN2 = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1)  

        self.flatten = nn.Flatten()

    def prePrepare(self, valueStock):
        """
        Convert stock sheets into binary tensor.
        """
        stocks_np = np.stack(valueStock)  
        stocks_binary = (stocks_np == -1).astype(np.float32)
        binary_matric_stocks = torch.from_numpy(stocks_binary).unsqueeze(1)  
        return binary_matric_stocks

    def decodeToVector(self, valueStock, ProductCount):
        binary_matric_stocks = self.prePrepare(valueStock).to(next(self.parameters()).tool)
        outcomeCnn = []
        for i in range(self.total_stock):
            sheet = binary_matric_stocks[i].unsqueeze(0) 
            sheet = F.relu(self.cnnN1(sheet))  
            sheet = F.relu(self.cnnN2(sheet))  
            sheet = self.flatten(sheet)  
            outcomeCnn.append(sheet)
        cnn_out = torch.cat(outcomeCnn, dim=1)  

        featuresP = []
        for product in ProductCount:
            length, width = product["size"]
            quantity = product["quantity"]
            featuresP.append([length, width, quantity])
        while len(featuresP) < self.total_product:
            featuresP.append([0.0, 0.0, 0.0])
        featuresP = featuresP[:self.total_product]

        featuresP = np.array(featuresP, dtype=np.float32)
        featuresP = torch.from_numpy(featuresP).view(1, -1).to(next(self.parameters()).tool)
        mergerInput = torch.cat([cnn_out, featuresP], dim=1)  
        return mergerInput


class ActorNetwork(nn.Module, Policy):
    def __init__(self, total_stock, High_Weight, High_Height, total_product, max_in_one, hiddenD):
        super(ActorNetwork, self).__init__()
        self.total_stock = total_stock
        self.High_Weight = High_Weight
        self.High_Height = High_Height
        self.total_product = total_product
        self.max_in_one = max_in_one
        self.hiddenD = hiddenD

        self.encoder = VectorStateE(total_stock, High_Weight, High_Height, total_product, max_in_one)

        outputCnn = 4 * total_stock * (High_Weight // 4) * (High_Height // 4)  
        outputProduct = 3 * total_product
        input_hidden = outputCnn + outputProduct

        self.fullyConnected1 = nn.Linear(input_hidden, hiddenD)
        self.fullyConnected2 = nn.Linear(hiddenD, 2 * total_stock * total_product)

    def CheckProductPlace(self, stock, position, size):
        """
        Checks if a product can be placed at a specific position on the stock.
        """
        x, y = position
        w, h = size

        # Convert stock to a torch.Tensor if it's a numpy array
        if isinstance(stock, np.ndarray):
            stock = torch.tensor(stock, dtype=torch.int32)

        # Perform the check using torch.all
        return torch.all(stock[x:x + w, y:y + h] == -1)

    def get_action(self, ProductCount, valueStock):
        """
        Returns a tensor where valid actions are marked as True.
        """
        productss = len(ProductCount)
        actionOk = torch.zeros(2 * self.total_stock * self.total_product, dtype=torch.bool).to(
            next(self.parameters()).tool)

        for sheet in range(self.total_stock):
            for product_idx in range(productss):
                if product_idx < self.total_product and ProductCount[product_idx]["quantity"] > 0:
                    actionIndex = sheet * self.total_product + product_idx
                    for m in range(2):  
                        action_index = actionIndex * 2 + m
                        product_info = ProductCount[product_idx]
                        prod_w, prod_h = product_info["size"]

                        if m == 0:
                            for i in range(self.High_Weight - prod_w + 1):
                                for j in range(self.High_Height - prod_h + 1):
                                    if self.CheckProductPlace(valueStock[sheet], (i, j), (prod_w, prod_h)):
                                        actionOk[action_index] = True
                                        break
                            if actionOk[action_index]:
                                break  

                        elif m == 1:
                            prod_w, prod_h = prod_h, prod_w  
                            for i in range(self.High_Weight - prod_w + 1):
                                for j in range(self.High_Height - prod_h + 1):
                                    if self.CheckProductPlace(valueStock[sheet], (i, j), (prod_w, prod_h)):
                                        actionOk[action_index] = True
                                        break
                            if actionOk[action_index]:
                                break

        return actionOk

    def decodeToVector(self, valueStock, ProductCount):
        mergerInput = self.encoder(valueStock, ProductCount)
        x = F.relu(self.fullyConnected1(mergerInput))
        actionL = self.fullyConnected2(x)  

        actionOk = self.get_action(ProductCount, valueStock)

        if actionL.dim() == 2:
            actionOk = actionOk.unsqueeze(0).expand(actionL.size(0), -1)

        actionL[~actionOk] = -1e9  
        probabilityAction = F.softmax(actionL, dim=1)

        return probabilityAction


class CriticNetwork(nn.Module):
    def __init__(self, total_stock, High_Weight, High_Height, total_product, max_in_one, hiddenD):
        super(CriticNetwork, self).__init__()
        self.total_stock = total_stock
        self.High_Weight = High_Weight
        self.High_Height = High_Height
        self.total_product = total_product
        self.max_in_one = max_in_one
        self.hiddenD = hiddenD

        self.encoder = VectorStateE(total_stock, High_Weight, High_Height, total_product, max_in_one)

        outputCnn = 4 * total_stock * (High_Weight // 4) * (High_Height // 4)
        outputProduct = 3 * total_product
        input_hidden = outputCnn + outputProduct

        self.fullyConnected1 = nn.Linear(input_hidden, hiddenD)
        self.fullyConnected2 = nn.Linear(hiddenD, hiddenD)
        self.fullyConnected3 = nn.Linear(hiddenD, 1)

    def decodeToVector(self, valueStock, ProductCount):
        mergerInput = self.encoder(valueStock, ProductCount)
        x = F.relu(self.fullyConnected1(mergerInput))
        x = F.relu(self.fullyConnected2(x))
        value = self.fullyConnected3(x)

        print(f"Critic Value: {value}")

        return value.squeeze(1)

class A2CAgent(Policy):
    def __init__(self, tool, total_stock, High_Weight, High_Height, total_product, max_in_one, hiddenD, learning=0.001,
                 gamma=0.9, epsilon=1.0, epsilonMin=0.1, epsilonDecay=0.995):
        super(A2CAgent, self).__init__()
        self.tool = tool
        self.total_stock = total_stock
        self.High_Weight = High_Weight
        self.High_Height = High_Height
        self.total_product = total_product
        self.max_in_one = max_in_one
        self.hiddenD = hiddenD
        self.learning = learning
        self.gamma = gamma
        self.epsilon = epsilon  
        self.epsilonMin = epsilonMin  
        self.epsilonDecay = epsilonDecay  

        self.actor = ActorNetwork(total_stock, High_Weight, High_Height, total_product, max_in_one, hiddenD).to(
            self.tool)
        self.critic = CriticNetwork(total_stock, High_Weight, High_Height, total_product, max_in_one, hiddenD).to(
            self.tool)
        self.optimizeActor = optim.Adam(self.actor.parameters(), learning=self.learning)
        self.optim_critic = optim.Adam(self.critic.parameters(), learning=self.learning)

        self.log_probability = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def chooseAction(self, stockOrProduct):
        valueStock = stockOrProduct["stocks"]
        ProductCount = stockOrProduct["products"]

        probabilityAction = self.actor(valueStock, ProductCount)
        distribution = torch.distributions.Categorical(probabilityAction)

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            choosedAction = distribution.sample()  
        else:
            choosedAction = torch.argmax(probabilityAction, dim=1)  

        log_prob = distribution.log_prob(choosedAction)
        entropy = distribution.entropy()

        self.log_probability.append(log_prob)
        self.values.append(self.critic(valueStock, ProductCount))
        self.entropies.append(entropy)

        action_index = choosedAction.item()
        sheet_index = action_index // (2 * self.total_product)
        product_index = (action_index % (2 * self.total_product)) // 2
        rotation = action_index % 2  

        return sheet_index, product_index, rotation

    def place_product(self, stockOrProduct, sheet_index, product_index, rotation):
        stock = stockOrProduct["stocks"][sheet_index]
        prod_info = stockOrProduct["products"][product_index]
        prod_w, prod_h = prod_info["size"]
        if rotation == 1:
            prod_w, prod_h = prod_h, prod_w  

        High_Weight, High_Height = stock.shape
        nicePositionX, nicePositionY = None, None

        for positionX in range(High_Weight - prod_w + 1):
            for positionY in range(High_Height - prod_h + 1):
                if self.CheckProductPlace(stock, (positionX, positionY), (prod_w, prod_h)):
                    nicePositionX, nicePositionY = positionX, positionY
                    break
            if nicePositionX is not None:
                break

        if nicePositionY is None:
            return None

        return {
            "stock_idx": sheet_index,
            "size": [prod_w, prod_h],
            "position": [nicePositionX, nicePositionY],
        }

    def finalAction(self, stockOrProduct, info):
        choosedAction = self.chooseAction(stockOrProduct)
        sheet_idx, product_idx, rotation = choosedAction
        placed = self.place_product(stockOrProduct, sheet_idx, product_idx, rotation)
        return placed

    def newUpdate(self, done):
        if len(self.rewards) == 0:
            return

        assert len(self.rewards) == len(self.log_probability) == len(self.values) == len(self.entropies)
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).squeeze(1).to(self.tool)

        returnMean = returns.mean()
        returnStd = returns.std(unbiased=False)
        returns = (returns - returnMean) / (returnStd + 1e-5)

        values = torch.stack(self.values).squeeze(1).to(self.tool)
        log_probability = torch.stack(self.log_probability).to(self.tool)
        entropies = torch.stack(self.entropies).to(self.tool)

        advantage = returns - values

        lossedActor = -(log_probability * advantage.detach()).mean() - 0.1 * entropies.mean()
        lossedCritic = advantage.pow(2).mean()

        self.optimizeActor.zero_grad()
        lossedActor.backward()
        self.optimizeActor.step()

        self.optim_critic.zero_grad()
        lossedCritic.backward()
        self.optim_critic.step()

        self.log_probability = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def storedAward(self, reward):
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.tool))

    def calReward(self, stockOrProduct, done, C=1):
        totalWaste = 0
        totalArea = 0
        countStock = 0
        if not done:
            print(f"Reward: 0")
            return 0
        for idx, sheet in enumerate(stockOrProduct["stocks"]):
            if np.any(sheet >= 0):
                waste = np.sum(sheet == -1)
                area = np.sum(sheet != -2)
                totalWaste += waste
                totalArea += area
                countStock += 1

        waste_ratio = totalWaste / totalArea if totalArea > 0 else 0

        reward = C / waste_ratio  # Penalty for waste

        print(f"Reward: {reward}")
        return reward

    def CheckProductPlace(self, stock, position, size):
        x, y = position
        w, h = size
        return np.all(stock[x:x + w, y:y + h] == -1)

    def saveModel(self, filepath):
        torch.save(self.actor.state_dict(), filepath + '_actor.pth')
        torch.save(self.critic.state_dict(), filepath + '_critic.pth')
        print(f"Actor and Critic models saved to {filepath}_actor.pth and {filepath}_critic.pth.")

    def loadModel(self, filepath, train=True):
        actorPath = filepath + '_actor.pth'
        criticPath = filepath + '_critic.pth'

        if not os.path.exists(actorPath) or not os.path.exists(criticPath):
            print(f"Error: Model files not found. Initializing model from scratch.")
            self.initialModel()
            return

        try:
            self.actor.load_state_dict(torch.load(actorPath))
            self.critic.load_state_dict(torch.load(criticPath))
            print(f"Models loaded from {actorPath} and {criticPath}.")
            if not train:
                self.actor.eval()
                self.critic.eval()
            else:
                self.actor.train()
                self.critic.train()

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.initialModel()

    def initialModel(self):
        print("Initializing Actor and Critic networks...")
        self.actor = ActorNetwork(self.total_stock, self.High_Weight, self.High_Height, self.total_product,
                                  self.max_in_one, self.hiddenD).to(self.tool)
        self.critic = CriticNetwork(self.total_stock, self.High_Weight, self.High_Height, self.total_product,
                                    self.max_in_one, self.hiddenD).to(self.tool)

        self.optimizeActor = optim.Adam(self.actor.parameters(), learning=self.learning)
        self.optim_critic = optim.Adam(self.critic.parameters(), learning=self.learning)

        self.actor.train()
        self.critic.train()

        print("Actor and Critic networks initialized from scratch. Starting training...")

    def close(self):
        pass