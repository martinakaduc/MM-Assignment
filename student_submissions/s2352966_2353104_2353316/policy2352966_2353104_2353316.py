from policy import Policy
import numpy as np


class Policy2352966_2353104_2353316(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id

    def get_action(self, observation, info):
        # Student code here
        return self.action1(observation) if self.policy_id == 1 else self.action2(observation)

    def action1(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Sort products in descending order by area
        sortedProducts = sorted(products, key = lambda x: x["size"][0] * x["size"][1], reverse = True)
        
        # Sort stocks in descending order by area
        sortedStocks = sorted(zip(stocks, range(len(stocks))), key = lambda x: self._get_stock_size_(x[0])[0] * self._get_stock_size_(x[0])[1], reverse = True)

        for product in sortedProducts:
            if product["quantity"] < 1:
                continue
            productSize = product["size"]
            for stock, stockIdx in sortedStocks:
                stockW, stockH = self._get_stock_size_(stock)
                # If cannot place, rotate the product to try
                if stockW < productSize[0] or stockH < productSize[1]:
                    productSize = productSize[::-1]
                # Skip if still cannot place   
                if stockW < productSize[0] or stockH < productSize[1]:
                    continue
                        
                for w in range(stockW - productSize[0] + 1):
                    for h in range(stockH - productSize[1] + 1):
                        position = (w, h)
                        if not self._can_place_(stock, position, productSize):
                            continue
                        return {
                            "stock_idx": stockIdx,
                            "size": productSize,
                            "position": position
                        }
                
        # Return nothing
        return {
            "stock_idx": -1,
            "size": [0, 0],
           "position": (0, 0)
        }
        
    def action2(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Sort products in descending order by area
        sortedProducts = sorted(products, key = lambda x: x["size"][0] * x["size"][1], reverse = True)
        
        for product in sortedProducts:
            if product["quantity"] < 1:
                continue
            bestFit = None
            bestStockIdx = -1
            for stockIdx, stock in enumerate(stocks):
                fit = self.findBestFit(stock, product["size"])
                if fit is None:
                    continue
                position, rotated = fit
                # Chose the position with the least wasted area
                if bestFit is None or self.wastedArea(stock, product["size"] if not rotated else product["size"][::-1], position) < self.wastedArea(stock, product["size"] if not bestFit[1] else product["size"][::-1], bestFit[0]):
                    bestFit = (position, rotated)
                    bestStockIdx = stockIdx
            position, rotated = bestFit
            
            # Return action
            return {
                "stock_idx": bestStockIdx,
                "size": product["size"] if not rotated else product["size"][::-1],
                "position": position
            }            
            
    # Find the best fit position and rotation for the product in a stock
    def findBestFit(self, stock, productSize):
        stockW, stockH = self._get_stock_size_(stock)
        bestPosition = None
        
        # Check unrotated placement
        for w in range(stockW - productSize[0] + 1):
            for h in range(stockH - productSize[1] + 1):
                if not self._can_place_(stock, (w, h), productSize):
                    continue
                return (w, h), False
        
        # Check rotated placement if unrotated does not fit 
        productSize = productSize[::-1]
        for w in range(stockW - productSize[0] + 1):
            for h in range(stockH - productSize[1] + 1):
                if not self._can_place_(stock, (w, h), productSize):
                    continue
                return (w, h), True
        
        return None
    
    # Calculate the wasted area around the placed product
    def wastedArea(self, stock, productSize, position):
        w, h = position
        stockW, stockH = self._get_stock_size_(stock)
        
        # Calculate unused area around the product
        unusedAbove = h
        unusedBelow = stockH - (h + productSize[1])
        unusedLeft = w
        unusedRight = stockW - (w + productSize[0])
        
        return unusedAbove + unusedBelow + unusedLeft + unusedRight
        
        #
        
    # Student code here
    # You can add more functions if needed
