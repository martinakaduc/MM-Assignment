from policy import Policy

# import additional libraries
from abc import abstractmethod
import numpy as np

# CLASS POLICY
class Policy2352026_2352808_2352937_2352973_2353127(Policy):
    def __init__(self, policy_id=1):
        if policy_id not in [1, 2]:
            raise ValueError("Policy ID must be 1 or 2")

        self.policy_id = policy_id
        # Instantiate the correct policy object based on the policy_id
        self.policy = GreedyPolicy() if policy_id == 1 else FFDPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)


# CLASS GREEDY POLICY
class GreedyPolicy(Policy):
    def __init__(self):
        self.sortedStockindex = np.array([])
        self.sortedProds = []
        self.sortCount = 0

    def sortStockandProduct(self, stockList, prodList):
        # Compute the area for each stock and sort stocks accordingly
        stock_areas = [self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for stock in stockList]
        # Reverse sorting order to get largest stock first
        self.sortedStockindex = np.argsort(stock_areas)[::-1]
        # Sort products based on their size (area)
        self.sortedProds = sorted(prodList, key=lambda prod: prod['size'][0] * prod['size'][1], reverse=True)

    def get_action(self, observation, info):
        # Initialize variables for selected product and stock
        selectProdSize = [0, 0]
        selectStockindex = -1
        posX, posY = None, None

        if self.sortCount == 0:
            self.sortStockandProduct(observation['stocks'], observation['products'])
        self.sortCount += 1

        # Reset sorting after all products have been placed
        if self.sortedProds[-1]['quantity'] == 1:
            self.sortCount = 0  # Reset sortCount

        # Attempt to place the products in available stocks
        for product in self.sortedProds:
            if product["quantity"] > 0:
                selectProdSize = product["size"]

                for stock_idx in self.sortedStockindex:
                    stockWidth, stockHeight = self._get_stock_size_(observation['stocks'][stock_idx])
                    productWidth, productHeight = selectProdSize

                    # Skip if product doesn't fit in the stock
                    if stockWidth < productWidth or stockHeight < productHeight:
                        continue

                    # Try finding a position within the stock
                    posX, posY = None, None
                    for x in range(stockWidth - productWidth + 1):
                        for y in range(stockHeight - productHeight + 1):
                            if self._can_place_(observation['stocks'][stock_idx], (x, y), selectProdSize):
                                posX, posY = x, y
                                break
                        if posX is not None and posY is not None:
                            break

                    if posX is not None and posY is not None:
                        selectStockindex = stock_idx
                        break

                if posX is not None and posY is not None:
                    break

        return {
            "stock_idx": selectStockindex,
            "stock_size": self._get_stock_size_(observation['stocks'][selectStockindex]),
            "size": selectProdSize,
            "position": (posX, posY)
        }

# CLASS FIRST FIT DECREASING POLICY
class FFDPolicy(Policy):
    def __init__(self):
        self.sortedStockindex = np.array([])  # Not necessary in FFD
        self.sortedProds = []
        self.sortCount = 0

    def sortStockandProduct(self, stockList, prodList):
        # Compute area and sort both stocks and products in descending order of area
        stock_areas = [self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for stock in stockList]
        self.sortedStockindex = np.argsort(stock_areas)[::-1]
        self.sortedProds = sorted(prodList, key=lambda prod: prod['size'][0] * prod['size'][1], reverse=True)

    def get_action(self, observation, info):
        # Initialize variables for selected product and stock
        selectProdSize = [0, 0]
        selectStockindex = -1
        posX, posY = None, None

        if self.sortCount == 0:
            self.sortStockandProduct(observation['stocks'], observation['products'])
        self.sortCount += 1

        # Reset if no products remain
        if self.sortedProds[-1]['quantity'] == 1:
            self.sortCount = 0  # Reset sortCount

        # Try placing products in stocks
        for product in self.sortedProds:
            if product["quantity"] > 0:
                selectProdSize = product["size"]

                # Check stocks in order to find the first one that fits
                for stock_idx in self.sortedStockindex:
                    stockWidth, stockHeight = self._get_stock_size_(observation['stocks'][stock_idx])
                    productWidth, productHeight = selectProdSize

                    # Skip if product doesn't fit
                    if stockWidth < productWidth or stockHeight < productHeight:
                        continue

                    posX, posY = None, None
                    # Check positions where the product can fit
                    for x in range(stockWidth - productWidth + 1):
                        for y in range(stockHeight - productHeight + 1):
                            if self._can_place_(observation['stocks'][stock_idx], (x, y), selectProdSize):
                                posX, posY = x, y
                                break
                        if posX is not None and posY is not None:
                            break

                    if posX is not None and posY is not None:
                        selectStockindex = stock_idx
                        break

                if posX is not None and posY is not None:
                    break

        return {
            "stock_idx": selectStockindex,
            "stock_size": self._get_stock_size_(observation['stocks'][selectStockindex]),
            "size": selectProdSize,
            "position": (posX, posY)
        }
