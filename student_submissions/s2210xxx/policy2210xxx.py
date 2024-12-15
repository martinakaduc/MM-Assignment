import numpy as np
from policy import Policy

class Policy2213486_2313771_2313334_2212896_2313431(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.__list_of_sorted_products_indexes = []
        self.__list_of_sorted_stocks_indexes = []
        self.__list_of_stocks_used = []
        self.__current_idx = 0
        self._total_of_quantities = 0
        self.__list_of_products_size = []
        self.__list_of_stocks_size = []

    def _reset_attributes(self):
        self.__list_of_sorted_products_indexes = []
        self.__list_of_sorted_stocks_indexes = []
        self.__list_of_stocks_used = []
        self.__current_idx = 0
        self._total_of_quantities = 0
        self.__list_of_products_size = []
        self.__list_of_stocks_size = []

    def _remaining_free_area(self, stock):
        """
        Empty area of stock.
        """
        return np.sum(stock == -1)

    def _sum_of_quantities(self, list_of_product):
        """
        Find sum of quantities of products
        """
        result = 0
        for prod in list_of_product:
            result += prod["quantity"]
        return result

    def _get_stock_area(self, stock):
        return np.sum(stock != -2)

    def _copy_arr_size(self, arr_size):
        result = []
        for i in range(len(arr_size)):
            result.append(arr_size[i][:])
        return result

    def _is_2_arr_size_same(self, arr1, arr2):
        if len(arr1) != len(arr2):
            return False
        for i in range(len(arr1)):
            a, b = arr1[i]
            c, d = arr2[i]
            if a != c or b != d:
                return False
        return True

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.first_fit_decreasing(observation)
        elif self.policy_id == 2:
            return self.best_fit_decreasing_height(observation)

    def first_fit_decreasing(self, observation):
        """
        Implements the First Fit Decreasing algorithm.
        """
        products = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],  # Sort by area (width * height)
            reverse=True
        )

        stocks = observation["stocks"]

        for product in products:
            prod_width, prod_height = product["size"]

            for stock_idx, stock in enumerate(stocks):
                stock_width, stock_height = self._get_stock_size_(stock)
                free_area = self._remaining_free_area(stock)
                if stock_width >= prod_width and stock_height >= prod_height and (prod_width * prod_height <= free_area):
                    for x in range(stock_width - prod_width + 1):
                        for y in range(stock_height - prod_height + 1):
                            if self._can_place_(stock, (x, y), (prod_width, prod_height)):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": (prod_width, prod_height),
                                    "position": (x, y)
                                }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def best_fit_decreasing_height(self, observation):
        """
        Implements the Best Fit Decreasing Height algorithm.
        """
        products = observation["products"]
        list_of_products_size = [prod["size"] for prod in products]
        stocks = observation["stocks"]
        list_of_stocks_size = [self._get_stock_size_(stock) for stock in stocks]

        if (not self._is_2_arr_size_same(list_of_products_size, self.__list_of_products_size) or
                not self._is_2_arr_size_same(list_of_stocks_size, self.__list_of_stocks_size) or
                self._total_of_quantities != self._sum_of_quantities(products)):
            if not self.__list_of_products_size and not self.__list_of_stocks_size:
                self.__list_of_products_size = self._copy_arr_size(list_of_products_size)
                self.__list_of_stocks_size = self._copy_arr_size(list_of_stocks_size)
            else:
                self._reset_attributes()

        if not self.__list_of_sorted_products_indexes:
            self.__list_of_sorted_products_indexes = sorted(
                [i for i, product in enumerate(products)],
                key=lambda i: products[i]["size"][0] * products[i]["size"][1], reverse=True
            )
            self._total_of_quantities += sum(product["quantity"] for product in products)

        if not self.__list_of_sorted_stocks_indexes:
            self.__list_of_sorted_stocks_indexes = sorted(
                [i for i, stock in enumerate(stocks)],
                key=lambda i: self._get_stock_size_(stocks[i])[0] * self._get_stock_size_(stocks[i])[1], reverse=True
            )

        idx = self.__list_of_sorted_products_indexes[self.__current_idx]
        sum_of_quantities = self._sum_of_quantities(products)

        count_val = 0
        while products[idx]["quantity"] == 0 and sum_of_quantities > 0:
            if count_val == len(self.__list_of_sorted_products_indexes):
                break
            self.__current_idx += 1
            if self.__current_idx >= len(self.__list_of_sorted_products_indexes):
                self.__current_idx = 0
            idx = self.__list_of_sorted_products_indexes[self.__current_idx]
            count_val += 1

        product = products[idx]
        product_size = product["size"]

        best_stock_idx = -1
        best_remain_area = float('inf')
        best_position = None
        best_orientation = None

        for stock_idx in self.__list_of_stocks_used:
            stock = stocks[stock_idx]
            stock_width, stock_height = self._get_stock_size_(stock)

            for orientation in [(product_size[0], product_size[1]), (product_size[1], product_size[0])]:
                width, height = orientation
                free_area = self._remaining_free_area(stock)
                if stock_width >= width and stock_height >= height and (width * height <= free_area):
                    for x in range(stock_width - width + 1):
                        for y in range(stock_height - height + 1):
                            if self._can_place_(stock, (x, y), orientation):
                                remaining_area = free_area - (width * height)
                                if remaining_area < best_remain_area:
                                    best_remain_area = remaining_area
                                    best_stock_idx = stock_idx
                                    best_position = (x, y)
                                    best_orientation = orientation

        if best_stock_idx != -1:
            self.__current_idx += 1
            if self.__current_idx >= len(self.__list_of_sorted_products_indexes):
                self.__current_idx = 0
            if self._sum_of_quantities(observation["products"]) == 1:
                self._reset_attributes()
            self._total_of_quantities -= 1
            return {
                "stock_idx": best_stock_idx,
                "size": best_orientation,
                "position": best_position
            }

        next_stock = len(self.__list_of_stocks_used)
        self.__list_of_stocks_used.append(next_stock)

        self.__current_idx += 1
        if self.__current_idx >= len(self.__list_of_sorted_products_indexes):
            self.__current_idx = 0
        if self._sum_of_quantities(observation["products"]) == 1:
            self._reset_attributes()
        self._total_of_quantities -= 1
        return {
            "stock_idx": next_stock,
            "size": product_size,
            "position": (0, 0)  }