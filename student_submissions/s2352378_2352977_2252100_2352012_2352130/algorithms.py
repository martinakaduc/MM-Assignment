from policy import Policy;
from student_submissions.s2352378_2352977_2252100_2352012_2352130.product_list import ProductList
from student_submissions.s2352378_2352977_2252100_2352012_2352130.stock import Stock;

INF = 1e9;


# ----------------------------------------------------------------
#   Branch and Bound Policy
# ----------------------------------------------------------------

# This policy uses two supporting classes: ProductList and Stock from stock.py file and product_list.py file respectively
class BranchAndBound(Policy):
    def __init__(self):
        self.list_product_actions = [];
        self.list_offset_actions = [];
        self.list_stock_actions = [];
        self.used_stock_list = [];
        self.placed_products = [];
        self.original_product_list = None;
        self.original_stock_list = None;

    def get_action(self, observation, info):        
        # If there are no action in the list then generate the list of actions
        if not self.list_product_actions or info["filled_ratio"] == 0.0:
            self.generate_list_actions(observation, info);

        # Get the action from the list
        stock_index = self.list_stock_actions.pop(0);
        product_size = observation["products"][self.list_product_actions.pop(0)]["size"];
        pos_x, pos_y = self.list_offset_actions.pop(0);
        
        return {"stock_idx": stock_index, "size": product_size, "position": (pos_x, pos_y)}


    # Generate list of actions
    def generate_list_actions(self, observation, info):
        # Reset policy if filled ratio is equal to 0.0
        if info["filled_ratio"] == 0.0:
            self.reset_policy();
        
        self.original_product_list = ProductList(observation["products"]);
        self.original_stock_list = observation["stocks"];
        
        stock = self.find_suitable_stock();
        
        wasted_area = max_wasted_area = self.find_max_wasted_area(stock, take_action=True);
        min_wasted_area = self.find_min_wasted_area(stock);
        
        if max_wasted_area != min_wasted_area:
            result = self.try_place_product(stock, max_wasted_area, min_wasted_area);
            
            wasted_area, (placed_products, placed_offsets) = result;
            
            if placed_products:
                self.list_product_actions = placed_products;
                self.list_offset_actions = placed_offsets;
        
        for i in range(len(self.list_product_actions)):
            self.list_stock_actions.append(stock.get_stock_index());
        
        # Reset product list
        for placed_product in self.placed_products:
            self.original_product_list.add_quantity(placed_product);
        self.placed_products = [];

    # Find initial maximum wasted area(upper bound) by trying to place all products into the stock
    def find_max_wasted_area(self, stock, take_action=True, product_list=None):
        stock_w, stock_h = stock.get_stock_wh();
        if product_list is None:
            product_list = self.original_product_list.get_mapped_product_list();
        
        pos_x, pos_y = 0, 0;
        max_pos_x = 0;
        sum_product = 0;
        
        while pos_x <= stock_w:
            out_of_product = True;
            for index, product in product_list:
                product_w, product_h = product["size"];
                
                if product["quantity"] <= 0:
                    continue;
                else:
                    out_of_product = False;
                
                if pos_y + product_h > stock_h:
                    pos_y = 0;
                    pos_x = max_pos_x;
                    break;
                    
                if pos_x + product_w > stock_w:
                    pos_x += product_w;
                    break;
                
                fit_count = min((stock_h - pos_y) // product_h, product["quantity"]);
                
                product["quantity"] -= fit_count;
                sum_product += product_w * product_h * fit_count;
                for j in range(fit_count):
                    if take_action:
                        self.list_product_actions.append(index);
                        self.list_offset_actions.append([pos_x, pos_y]);
                    pos_y += product_h;
                max_pos_x = max(max_pos_x, pos_x + product_w);
                
            if out_of_product:
                break;
        
        # If there is no product to place then the wasted area is equal to zero
        if sum_product == 0:
            return 0;
        return stock_h * stock_w - sum_product;
    
    # find the minimum wasted area (lower bound) by using First Fit Decreasing algorithm
    # convert the product and stock to area and solve with First Fit Decreasing algorithm
    def find_min_wasted_area(self, stock):
        stock_w, stock_h = stock.get_stock_wh();
        stock_area = stock_w * stock_h;
        total_product_area = 0;
        product_list = self.original_product_list.get_mapped_product_list(lambda x: (x[1]["size"][0] * x[1]["size"][1]), reverse=True);
        
        for product in product_list:
            if product[1]["quantity"] <= 0:
                continue;
            
            product_w, product_h = product[1]["size"];
            
            if product_w > stock_w or product_h > stock_h:
                continue;
            
            product_area = product_w * product_h;
            wasted_area = stock_area - total_product_area;
            
            fit_count = min(wasted_area // product_area, product[1]["quantity"]);
            
            total_product_area += fit_count * product_area;
            
        return stock_area - total_product_area;
    
    # Find suitable stock to place products
    # Calculate 2 steps ahead with the find_max_wasted_area method to find the best stock to place products
    def find_suitable_stock(self):
        stock_list = self.original_stock_list;
        suitable_stock = None;
        total_wasted_area = INF;
        
        # Loop through all stocks to find the lowest max wasted area of the two chosen stocks
        for i in range(len(stock_list)):
            # Check if stock is used
            if i in self.used_stock_list:
                continue;
            
            product_list1 = self.original_product_list.get_mapped_product_list();
            stock1 = Stock(stock_list[i], i, self.original_product_list);
            new_wasted_area1 = self.find_max_wasted_area(stock1, False, product_list1);
            
            # Skip if the wasted area is greater than the total wasted area
            if new_wasted_area1 < total_wasted_area:
                for j in range(len(stock_list)):
                    # Check if stock is used
                    if j in self.used_stock_list:
                        continue;
                    
                    stock2 = Stock(stock_list[j], j, self.original_product_list);
                    product_list2 = self.original_product_list.get_deep_copy(product_list1)
                    new_wasted_area2 = self.find_max_wasted_area(stock2, False, product_list2);
                        
                    if new_wasted_area1 + new_wasted_area2 < total_wasted_area:
                        suitable_stock = stock1
                        total_wasted_area = new_wasted_area1 + new_wasted_area2;
            
        # Add the stock to the used stock list
        self.used_stock_list.append(suitable_stock.get_stock_index());
        return suitable_stock;

    # Try to place the product into the stock
    def try_place_product(self, stock, max_wasted_area=INF, min_wasted_area=0, product_cut_index=-1):
        stock_wh = stock.get_stock_wh();
        wasted_area = stock_wh[0] * stock_wh[1];
        max_wasted_area = min(wasted_area, max_wasted_area);
        result_placed_product = [[], []];
        
        # Align product to the width-or-height-fitted stock
        if product_cut_index != -1:
            product_wh = self.original_product_list.get_size(product_cut_index);
            offset = stock.get_offset();
            axis = stock.get_cut_axis();
            reverse_axis = 1 - axis;
            
            # Find the maximum product that could be fitted into the stock
            fit_count = min(stock_wh[axis] // product_wh[axis], self.original_product_list.get_quantity(product_cut_index));
            
            self.original_product_list.remove_quantity(product_cut_index, fit_count);
            for j in range(fit_count):
                self.placed_products.append(product_cut_index);
                result_placed_product[0].append(product_cut_index);
                result_placed_product[1].append(offset);
                array_offset = [0, 0];
                array_offset[axis] += product_wh[axis];
                offset = (offset[0] + array_offset[0], offset[1] + array_offset[1]);
                
            # Wasted area = 0
            if stock_wh[axis] / product_wh[axis] == fit_count:
                return 0, result_placed_product;
            
            new_cut = (product_wh[axis] * fit_count, reverse_axis);
            
            stock1, stock2 = stock.create_child(new_cut);
            
            # Try to place the product into the remaining stock
            new_wasted_area, new_result_placed_product = self.try_place_product(stock2, max_wasted_area, min_wasted_area, -1);
            
            result_placed_product[0] += new_result_placed_product[0];
            result_placed_product[1] += new_result_placed_product[1];
            
            return new_wasted_area, result_placed_product;
        
        possible_cuts, product_cut_indexes, perfect_fit = stock.get_possible_cuts();
                
        placed_product_count = len(self.placed_products);
        
        # If there is a perfect fit then place the product into the stock
        if perfect_fit:
            # Reduce the quantity of the product after placing it into the stock
            self.placed_products.append(perfect_fit[0]);
            self.original_product_list.remove_quantity(perfect_fit[0]);
            
            result_placed_product[0].append(perfect_fit[0]);
            result_placed_product[1].append(stock.get_offset());
            return 0, result_placed_product;
        
        # If there is no possible cut then return wasted area
        if not possible_cuts:
            return stock_wh[0] * stock_wh[1], result_placed_product;
        
        # Try to iterate all possible cuts
        for index, cut_info in enumerate(possible_cuts):
            stock1, stock2 = stock.create_child(cut_info);
            
            # Backtracking (add back the quantity of the product)
            while len(self.placed_products) > placed_product_count:
                self.original_product_list.add_quantity(self.placed_products.pop());
                
            max_wasted_area = min(wasted_area, max_wasted_area);
            
            # Find the minimum wasted area of the first stock
            stock1_wasted_area, stock1_placed_result = self.try_place_product(stock1, max_wasted_area, 0, product_cut_indexes[index]);
            
            # If the stock1 wasted area is greater than the maximum wasted area then fathom the branch stock2
            if stock1_wasted_area < max_wasted_area:
                # the new maximum wasted area = maximum wasted area - stock1 wasted area, which is used for the upper bound of stock2
                stock2_wasted_area, stock2_placed_result = self.try_place_product(stock2, max_wasted_area - stock1_wasted_area, max(0, min_wasted_area - stock1_wasted_area), -1);
                
                if stock1_wasted_area + stock2_wasted_area < max_wasted_area:
                    # Update the maximum wasted area
                    wasted_area = stock1_wasted_area + stock2_wasted_area;
                    
                    result_placed_product = [stock1_placed_result[0] + stock2_placed_result[0], stock1_placed_result[1] + stock2_placed_result[1]];
            
            # If the wasted area is less than the lower bound then break the loop and return the result
            if wasted_area <= min_wasted_area:
                break;
            
        return wasted_area, result_placed_product;

    def reset_policy(self):
        self.list_product_actions = [];
        self.list_offset_actions = [];
        self.list_stock_actions = [];
        self.used_stock_list = [];
        self.original_product_list = None;
        self.original_stock_list = None;
        self.placed_products = [];


# ----------------------------------------------------------------
#   First Fit Decreasing Height Policy
# ----------------------------------------------------------------


class FirstFitDecreasingHeight(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        # Sort products by area in decreasing order
        products = sorted(
            observation["products"],
            key=lambda p: p["size"][1],
            reverse=True
        )

        for product in products:
            if product["quantity"] > 0:
                prod_size = product["size"]

                # Try to fit into stocks
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Check original orientation
                    if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                        pos = self._find_position(stock, prod_size)
                        if pos is not None:
                            return {"stock_idx": stock_idx, "size": prod_size, "position": pos}

                    # Check rotated orientation
                    if stock_w >= prod_size[1] and stock_h >= prod_size[0]:
                        rotated_size = (prod_size[1], prod_size[0])
                        pos = self._find_position(stock, rotated_size)
                        if pos is not None:
                            return {"stock_idx": stock_idx, "size": rotated_size, "position": pos}

        # If no placement is found, return a fallback action
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _find_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Check all positions in the stock
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)

        return None