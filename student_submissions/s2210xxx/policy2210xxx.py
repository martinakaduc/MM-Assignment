from policy import Policy


class MyPolicy(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.sorted_products = None  
        if policy_id == 1:
            self.get_action = self.get_action_1
        elif policy_id == 2:
            self.get_action = self.get_action_2


    def get_action_2(self, observation, info):
        # Add your implementation for policy 2 here
        list_prods = observation["products"]
        stocks = observation["stocks"]
        stock_idx = -1  
        prod_size = [0, 0]  
        pos_x, pos_y = 0, 0  
        used_stocks = set()
        products_to_pack = [prod for prod in list_prods if prod["quantity"] > 0]
        products_to_pack.sort(key=lambda x: max(x["size"]), reverse=False)  
        stocks_with_size = [(stock_idx, stock) for stock_idx, stock in enumerate(stocks)]
        stocks_with_size.sort(key=lambda x: (self._get_stock_size_(x[1])[0], self._get_stock_size_(x[1])[1]), reverse=True)

        for prod in products_to_pack:
            prod_size = prod["size"]
            for stock_idx, stock in stocks_with_size:
                stock_w, stock_h = self._get_stock_size_(stock)  
                prod_w, prod_h = prod_size  
                if stock_w < prod_w or stock_h < prod_h:
                    continue 
                found_pos = False 
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):  
                            pos_x, pos_y = x, y
                            found_pos = True  
                            if stock_idx not in used_stocks:
                                used_stocks.add(stock_idx)
                            break
                    if found_pos:
                        break  
                
                if found_pos:  
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    

    def get_action_1(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]
        used_stocks = set()

        products_to_pack = [prod for prod in list_prods if prod["quantity"] > 0]
        products_to_pack.sort(key=lambda x: max(x["size"]), reverse=True) 

        for prod in products_to_pack:
            prod_size = prod["size"]
            best_stock_idx = -1
            best_remaining_space = float("inf")  
            pos_x, pos_y = -1, -1

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                if stock_w < prod_w or stock_h < prod_h:
                    continue

                found_pos = False

                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            remaining_space = (stock_w - prod_w) * (stock_h - prod_h)  
                            if remaining_space < best_remaining_space:
                                best_stock_idx = stock_idx
                                best_remaining_space = remaining_space
                                pos_x, pos_y = x, y
                                found_pos = True
                            break
                    if found_pos:
                        break

            if best_stock_idx != -1:
                return {
                    "stock_idx": best_stock_idx,
                    "size": prod_size,
                    "position": (pos_x, pos_y)
                }

        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}




    def _calculate_remaining_space(self, stock, position, prod_size):
        """Calculate remaining space considering the placement position."""
        x, y = position
        prod_w, prod_h = prod_size

        stock_w, stock_h = self._get_stock_size_(stock)

        if x + prod_w > stock_w or y + prod_h > stock_h:
            return float('inf')  

        used_space = prod_w * prod_h
        total_space = stock_w * stock_h
        return total_space - used_space

    def SortProduct(self, list_product):
        return sorted(list_product, key=lambda x: x['size'][0] * x['size'][1], reverse=True)
