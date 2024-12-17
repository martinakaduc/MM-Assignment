from policy import Policy


class Policy2311051_2352890_2310467_2314019_2312054(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = 1
            self.index_prods = []
            self.index_stocks = []
            pass
        elif policy_id == 2:
            self.policy_id = 2
            self.index_prods = []
            self.index_stocks = []
            self.used_stocks = []
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            list_prods = observation["products"]
            list_stocks = observation["stocks"]
            # sort index of prod and stock
        
            if info["filled_ratio"] == 0:
                self.index_prods = [(prod["size"], i) for i, prod in enumerate(list_prods)]
                self.index_stocks = [(self._get_stock_size_(stock), i) for i, stock in enumerate(list_stocks)]
                self.index_prods = sorted(self.index_prods, key=lambda x: x[0][0] * x[0][1], reverse=True)
                self.index_stocks = sorted(self.index_stocks, key=lambda x: x[0][0] * x[0][1], reverse=False)

            # choose product
            while list_prods[self.index_prods[0][1]]["quantity"] == 0:
                self.index_prods.pop(0)

            for index_prod in self.index_prods:
                prod = list_prods[index_prod[1]]
            
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    for index_stock in self.index_stocks:
                        stock = list_stocks[index_stock[1]]
                        stock_w, stock_h = self._get_stock_size_(stock)

                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        return {
                                            "stock_idx": index_stock[1],
                                            "size": prod_size,
                                            "position": (x, y),
                                        }

                        if stock_w >= prod_h and stock_h >= prod_w:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        prod_size = prod_size[::-1]            
                                        return {
                                            "stock_idx": index_stock[1],
                                            "size": prod_size,
                                            "position": (x, y),
                                        }
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        else:
            list_prods = observation["products"]
            list_stocks = observation["stocks"]
            # sort index of prod and stock
        
            if info["filled_ratio"] == 0:
                self.index_prods = [(prod["size"], i) for i, prod in enumerate(list_prods)]
                self.index_stocks = [(self._get_stock_size_(stock), i) for i, stock in enumerate(list_stocks)]
                self.index_prods = sorted(self.index_prods, key=lambda x: x[0][0] * x[0][1], reverse=True)
                self.index_stocks = sorted(self.index_stocks, key=lambda x: x[0][0] * x[0][1], reverse=False)
                self.used_stocks = []
                self.used_stocks.append({"size": self.index_stocks[0][0], "loc": (0, 0), "index": self.index_stocks[0][1]})
                self.index_stocks.pop(0)

            # choose product
            while list_prods[self.index_prods[0][1]]["quantity"] == 0:
                self.index_prods.pop(0)

            for index_prod in self.index_prods:
                prod = list_prods[index_prod[1]]
            
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    for used_stock in self.used_stocks:
                        stock_w, stock_h = used_stock["size"]
                        if (prod_w > prod_h and stock_w > stock_h) or (prod_w < prod_h and stock_w < stock_h):
                            if stock_w >= prod_w and stock_h >= prod_h:
                                return self.get_prod(used_stock, prod_size)
                            if stock_w >= prod_h and stock_h >= prod_w:
                                list_prods[index_prod[1]]["size"] = list_prods[index_prod[1]]["size"][::-1]
                                return self.get_prod(used_stock, list_prods[index_prod[1]]["size"])
                        else:
                            if stock_w >= prod_h and stock_h >= prod_w:
                                list_prods[index_prod[1]]["size"] = list_prods[index_prod[1]]["size"][::-1]
                                return self.get_prod(used_stock, list_prods[index_prod[1]]["size"])
                            if stock_w >= prod_w and stock_h >= prod_h:
                                return self.get_prod(used_stock, prod_size)
        
            if self.used_stocks:
                for index_prod in self.index_prods:
                    prod = list_prods[index_prod[1]]
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        prod_w, prod_h = prod_size
                    
                        for index_stock in self.index_stocks:
                            stock = list_stocks[index_stock[1]]
                            stock_w, stock_h = self._get_stock_size_(stock)

                            if stock_w >= prod_w and stock_h >= prod_h:
                                self.used_stocks.append({"size": index_stock[0], "loc": (0, 0), "index": index_stock[1]})
                                self.index_stocks.remove(index_stock)
                                return self.get_action(observation, info)
                        
                            if stock_w >= prod_h and stock_h >= prod_w:
                                self.used_stocks.append({"size": index_stock[0], "loc": (0, 0), "index": index_stock[1]})
                                self.index_stocks.remove(index_stock)
                                return self.get_action(observation, info)

            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    # Student code here
    # You can add more functions if needed

    def get_prod(self, used_stock, size):
        prod_w, prod_h = size
        stock_w, stock_h = used_stock["size"]
        x_stock, y_stock = used_stock["loc"]
        index = used_stock["index"]

        if stock_w - prod_w != 0:
            self.used_stocks.append({"size": (stock_w - prod_w, stock_h), "loc": (x_stock + prod_w, y_stock), "index": index})
        if stock_h - prod_h != 0:
            self.used_stocks.append({"size": (prod_w, stock_h - prod_h), "loc": (x_stock, y_stock + prod_h), "index": index})
        self.used_stocks.remove(used_stock)
        
        return {"stock_idx": index, "size": size, "position": (x_stock, y_stock)}

        
