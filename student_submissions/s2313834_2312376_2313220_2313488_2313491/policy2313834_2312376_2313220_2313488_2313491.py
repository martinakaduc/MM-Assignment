from policy import Policy

class Policy2313834_2312376_2313220_2313488_2313491(Policy):
    def __init__(self, policy_id = 1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here

        self.policy_id = policy_id
        self.stock_idx = 0
        self.prod_idx = 0
        self.sum_demands = 0
        self.rotated = False


    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.origin_move(observation, info) # origin first fit without rotation
        else:
            return self.rotate_move(observation, info) # first fit with rotation


    def reset(self):
        self.stock_idx = 0
        self.prod_idx = 0
        self.sum_demands = 0


    def origin_cut(self, stock, prod_width, prod_height):
        stock_height, stock_width = self._get_stock_size_(stock)
        # first fit heuristic
        for y in range(stock_height):
            for x in range(stock_width):
                if self._can_place_(stock, (x, y), (prod_width, prod_height)):
                    return (x, y)
        return None

    def origin_move(self, observation, info):
        # get list of stocks and products
        stocks = observation["stocks"]
        products = observation["products"]

        # check if env has been reset
        if info['filled_ratio'] == 0.0:
            self.reset()

        # debugging
        product_demands = [prod["quantity"] for prod in products]
        if self.sum_demands == sum(product_demands):
            self.prod_idx += 1
            # self.stock_idx += 1
        self.sum_demands = sum(product_demands)

        # sort products by size and quantity
        products = sorted(products, key=lambda p: p['quantity'], reverse=False)
        products = sorted(products, key=lambda p: p['size'][0] * p['size'][1], reverse=True)

        # loop through stocks and products
        # fill full stock then move to next stock
        for s_idx, stock in enumerate(stocks[self.stock_idx:], start=self.stock_idx):

            for product in products[self.prod_idx:]:
                while product['quantity'] > 0:
                    prod_size = product['size']
                    prod_width, prod_height = prod_size

                    # get position to cut product
                    cut_position = self.origin_cut(stock, prod_width, prod_height)
                    if cut_position:
                        pos_x, pos_y = cut_position
                        # product['quantity'] -= 1 # fill be calculated in env.step()
                        return {
                            'stock_idx': s_idx,
                            'size': prod_size, 
                            'position': (pos_x, pos_y)
                        }
                    # if no valid cut was found, move to next product
                    else:
                        break
                # when current product quantity is 0, move to next product
                self.prod_idx += 1

            # when current product quantity is 0, move to next product
            self.prod_idx = 0
            self.stock_idx += 1


    def rotate_cut(self, stock, prod_width, prod_height):
        stock_height, stock_width = self._get_stock_size_(stock)
        # first fit heuristic
        if self.rotated == False:
            for y in range(stock_height):
                for x in range(stock_width):
                    if self._can_place_(stock, (x, y), (prod_width, prod_height)):
                        return (x, y)
                
        self.rotated = True
        # rotate product
        for y in range(stock_height):
            for x in range(stock_width):
                if self._can_place_(stock, (x, y), (prod_height, prod_width)):
                    return (x, y)
        
        return None

    def rotate_move(self, observation, info):
        # get list of stocks and products
        stocks = observation["stocks"]
        products = observation["products"]

        # check if env has been reset
        if info['filled_ratio'] == 0.0:
            self.reset()

        # debugging
        product_demands = [prod["quantity"] for prod in products]
        if self.sum_demands == sum(product_demands):
            self.prod_idx += 1
        self.sum_demands = sum(product_demands)

        # sort products by size and quantity
        products = sorted(products, key=lambda p: p['quantity'], reverse=False)
        products = sorted(products, key=lambda p: p['size'][0] * p['size'][1], reverse=True)

        # sort stocks by area
        # Create a list of tuples (stock, original_index)
        stock_with_index = [(stock, idx) for idx, stock in enumerate(stocks)]

        # Sort the list of tuples by the area of the stock
        stock_with_index.sort(key=lambda s: self._get_stock_size_(s[0])[0] * self._get_stock_size_(s[0])[1], reverse=True)

        # Extract the sorted stocks and their original indices
        stocks, original_indices = zip(*stock_with_index)


        # loop through stocks and products
        # fill full stock then move to next stock
        for s_idx, stock in enumerate(stocks[self.stock_idx:], start=self.stock_idx): 

            for p_idx, product in enumerate(products[self.prod_idx:], start=self.prod_idx):
                while product['quantity'] > 0:
                    prod_size = product['size']
                    prod_width, prod_height = prod_size

                    # get position to cut product
                    cut_position = self.rotate_cut(stock, prod_width, prod_height)
                    if cut_position:
                        pos_x, pos_y = cut_position
                        if self.rotated:
                            # print("rotated")
                            prod_size = prod_size[::-1]

                        # print("size :", prod_size)
                        # product['quantity'] -= 1 # fill be calculated in env.step()
                        return {
                            'stock_idx': original_indices[s_idx],
                            'size': prod_size, 
                            'position': (pos_x, pos_y)
                        }
                    # if no valid cut was found, move to next product
                    else:
                        self.rotated = False
                        break
                # when current product quantity is 0, move to next product
                self.prod_idx += 1

            # after iterate through all products, move to next stock, reset product index
            self.prod_idx = 0
            self.stock_idx += 1

