import random
from abc import abstractmethod
import numpy as np
from policy import Policy

class Policy2252344_1952705_2352032_2352636_2352142(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        if policy_id == 1:
            self.policy = GreedyAlgorithm()
        elif policy_id == 2:
            self.policy = BinPackingPolicysa()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class GreedyAlgorithm(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        products = observation["products"]
        selected_size = (0, 0)
        selected_stock_idx = -1
        selected_position = None

        for product in products:
            if product["quantity"] <= 0:
                continue

            original_size = tuple(product["size"])
            rotated_size = original_size[::-1]

            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_width, stock_height = self._get_stock_size_(stock)

                for size in (original_size, rotated_size):
                    product_width, product_height = size

                    if stock_width < product_width or stock_height < product_height:
                        continue

                    for x in range(stock_width - product_width + 1):
                        for y in range(stock_height - product_height + 1):
                            if self._can_place_(stock, (x, y), size):
                                selected_size = size
                                selected_stock_idx = stock_idx
                                selected_position = (x, y)
                                break
                        if selected_position:
                            break
                    if selected_position:
                        break

                if selected_position:
                    break

            if selected_position:
                break

        return {
            "stock_idx": selected_stock_idx,
            "size": selected_size,
            "position": selected_position,
        }


class BinPackingPolicysa(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        sorted_prods = sorted(
            (p for p in list_prods if p["quantity"] > 0),
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
        )

        for prod in sorted_prods:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                for orientation in [prod_size, prod_size[::-1]]:
                    ori_w, ori_h = orientation

                    for x in range(stock_w - ori_w + 1):
                        for y in range(stock_h - ori_h + 1):
                            if self._can_place_(stock, (x, y), orientation):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": orientation,
                                    "position": (x, y),
                                }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
