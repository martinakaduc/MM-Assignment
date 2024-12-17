from policy import Policy
import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans

class Policy2352837_2353272_2353038_2353305_2353358(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.get_action1(observation, info)
        elif self.policy_id == 2:
            return self.get_action2(observation, info)

    def _get_stock_size(self, stock):
        stock_width = np.sum(np.any(stock != -2, axis=1))
        stock_height = np.sum(np.any(stock != -2, axis=0))
        return stock_width, stock_height

    def _can_place(self, stock, position, product_size):
        pos_x, pos_y = position
        prod_width, prod_height = product_size
        return np.all(stock[pos_x:pos_x + prod_width, pos_y:pos_y + prod_height] == -1)

    def _find_bottom_left_position(self, stock, product_size, stock_width, stock_height):
        prod_width, prod_height = product_size
        best_position = None
        best_y = stock_height

        for pos_x in range(stock_width - prod_width + 1):
            for pos_y in range(stock_height - prod_height + 1):
                if self._can_place(stock, (pos_x, pos_y), product_size):
                    if best_position is None or pos_y < best_y or (pos_y == best_y and pos_x < best_position[0]):
                        best_position = (pos_x, pos_y)
                        best_y = pos_y
        return best_position

    def _find_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place(stock, (x, y), prod_size):
                    return x, y
        return -1, -1

    def get_action1(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        products = sorted(products, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        for product in products:
            if product["quantity"] > 0:
                product_size = product["size"]
                for stock_index, stock in enumerate(stocks):
                    stock_width, stock_height = self._get_stock_size(stock)

                    position = self._find_bottom_left_position(stock, product_size, stock_width, stock_height)
                    if position is not None:
                        return {"stock_idx": stock_index, "size": product_size, "position": position}

                    position = self._find_bottom_left_position(stock, product_size[::-1], stock_width, stock_height)
                    if position is not None:
                        return {"stock_idx": stock_index, "size": product_size[::-1], "position": position}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def get_action2(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]
        valid_prods = [prod for prod in list_prods if prod["quantity"] > 0]

        if not valid_prods:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        num_stocks = len(stocks)
        c = np.ones(num_stocks)
        A = []
        b = []

        for prod in valid_prods:
            prod_w, prod_h = prod["size"]
            row = []
            can_place = False

            for i, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size(stock)
                pos_x, pos_y = self._find_position(stock, (prod_w, prod_h))
                can_fit = (pos_x >= 0 and pos_y >= 0)
                row.append(1 if can_fit else 0)
                can_place |= can_fit

            if can_place:
                A.append(row)
                b.append(1)

        if not A:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        A = np.array(A)
        b = np.array(b)

        try:
            res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, 1), method='highs')
            if res.success:
                x = res.x
                selected_stock_idx = np.argmax(x)
                stock = stocks[selected_stock_idx]
                for prod in valid_prods:
                    pos_x, pos_y = self._find_position(stock, prod["size"])
                    if pos_x >= 0 and pos_y >= 0:
                        return {
                            "stock_idx": selected_stock_idx,
                            "size": prod["size"],
                            "position": (pos_x, pos_y)
                        }
        except:
            pass

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
