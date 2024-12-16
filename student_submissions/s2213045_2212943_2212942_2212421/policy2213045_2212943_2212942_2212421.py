from policy import Policy
import numpy as np
import random

class Policy2213045_2212943_2212942_2212421(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy_get_action = self.bin_packing_bfd_get_action
        elif policy_id == 2:
            self.policy_get_action = self.bin_packing_ffd_get_action

    def get_action(self, observation, info):
        """
        Unified method to call the appropriate policy's action.
        """
        return self.policy_get_action(observation, info)

    def bin_packing_bfd_get_action(self, observation, info):
        """
        Best Fit Decreasing (BFD) policy for bin packing.
        """
        sorted_prods = sorted(
            ({"product": p, "area": p["size"][0] * p["size"][1]} for p in observation["products"] if p["quantity"] > 0),
            key=lambda x: x["area"],
            reverse=True,
        )

        for prod_info in sorted_prods:
            prod = prod_info["product"]
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            best_fit = None

            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                for orientation in [prod_size, prod_size[::-1]]:
                    ori_w, ori_h = orientation

                    for x in range(0, stock_w - ori_w + 1, max(1, ori_w // 2)):
                        for y in range(0, stock_h - ori_h + 1, max(1, ori_h // 2)):
                            if self._can_place_(stock, (x, y), orientation):
                                remaining_area = (stock_w * stock_h) - (ori_w * ori_h)

                                if best_fit is None or remaining_area < best_fit["remaining_area"]:
                                    best_fit = {
                                        "stock_idx": stock_idx,
                                        "size": orientation,
                                        "position": (x, y),
                                        "remaining_area": remaining_area,
                                    }

            if best_fit:
                return {
                    "stock_idx": best_fit["stock_idx"],
                    "size": best_fit["size"],
                    "position": best_fit["position"],
                }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def bin_packing_ffd_get_action(self, observation, info):
        """
        First Fit Decreasing (FFD) policy for bin packing.
        """
        sorted_prods = sorted(
            ({"product": p, "area": p["size"][0] * p["size"][1]} for p in observation["products"] if p["quantity"] > 0),
            key=lambda x: x["area"],
            reverse=True,
        )

        for prod_info in sorted_prods:
            prod = prod_info["product"]
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
