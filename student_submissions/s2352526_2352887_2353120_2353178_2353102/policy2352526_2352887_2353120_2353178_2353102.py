from policy import Policy
import numpy as np


class Policy2352525_2352887_2353120_2353178_2353102(Policy):
    def __init__(self, policy_id = 2):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.stock_info_list = []
            self.stock_idx = []
            self.stock_sheets = []
            self.sorted_list_prods = []
            self.first_action = False
        elif policy_id == 2:
            self.inUsed_stocks = []  # List of used stocks
            self.notUsed_stocks = []  # List of unused stock    

    def SortProducts(self, products):
      return sorted(products, key=lambda x:-x["size"][0]*x["size"][1])

    def sort_stock_sheet(self, stocks):
        def calculate_stock_area(stock):
            width, height = self._get_stock_size_(stock)
            return width * height

        stock_info_with_indices = [
            (calculate_stock_area(stock), i, stock) for i, stock in enumerate(stocks)
        ]
        sorted_stocks = sorted(stock_info_with_indices, key=lambda x: x[0], reverse=True)

        self.stock_info_list = [(info[0], info[1], info[2]) for info in sorted_stocks]
        self.stock_idx = [info[1] for info in sorted_stocks]
        self.stock_sheets = [info[2] for info in sorted_stocks]

    def CanCut(self, stockSheet, pos, prod_size):
      return self._can_place_(stockSheet, pos, prod_size)

    def cut_product(self, product_size, stock_sheet):
        stock_width, stock_height = self._get_stock_size_(stock_sheet)
        product_width, product_height = product_size

        # Check if the product can fit in the stock sheet
        if product_width <= 0 or product_width > stock_width or product_height <= 0 or product_height > stock_height:
            return (-1, -1), product_size

        # Default values
        cut_position = (-1, -1)

        # Search for a suitable position to cut
        for x in range(stock_width - product_width + 1):
            for y in range(stock_height - product_height + 1):
                if self.CanCut(stock_sheet, (x, y), product_size):
                    cut_position = (x, y)
                    return cut_position, product_size

        return cut_position, product_size


    def get_action(self, observation, info):
        # Student code here
        # Policy 1
        if self.policy_id == 1:
            if not self.first_action or info.get("filled_ratio") == 0.0:
                self.first_action = True
                self.sort_stock_sheet(observation["stocks"])
                self.sorted_list_prods = self.SortProducts(observation["products"])

            cut_position = (-1, -1)
            product_size = None
            stock_idx = None

            for stock_index, stock_sheet in enumerate(self.stock_sheets):
                for product in self.sorted_list_prods:
                    if product["quantity"] > 0:
                        cut_position, product_size = self.cut_product(product["size"], stock_sheet)

                        if cut_position == (-1, -1):
                            cut_position, product_size = self.cut_product(product["size"][::-1], stock_sheet)

                        if cut_position != (-1, -1):
                            stock_idx = self.stock_idx[stock_index]
                            break

                if cut_position != (-1, -1):
                    break

            return {"stock_idx": stock_idx, "size": product_size, "position": cut_position}

        # Policy 2
        if self.policy_id == 2:
            # Reset when enviroment reset
            if not info["filled_ratio"]:
                self.inUsed_stocks = []
                self.notUsed_stocks = []

            if not self.notUsed_stocks:
                products_list = []
                for product in observation['products']:
                    area = product['size'][0] * product['size'][1]
                    products_list.append((area, product))

                self.sorted_products = [product for _, product in sorted(products_list, key=lambda x: x[0], reverse=True)]
                stocks_with_index = []
                for i, stock in enumerate(observation['stocks']):
                    # Kiểm tra xem stock là một dictionary hoặc numpy.ndarray
                    if isinstance(stock, dict) and 'stock' in stock:
                        stock_data = stock['stock']
                    elif isinstance(stock, np.ndarray):
                        stock_data = stock  
                    else:
                        stock_data = []

                    # Dùng hàm an toàn để xử lý dữ liệu
                    stock_size = self.safe_get_stock_size(stock_data)
                    stock_area = stock_size[0] * stock_size[1]
                    stocks_with_index.append({'index': i, 'stock': stock, 'area': stock_area})
                self.notUsed_stocks = sorted(stocks_with_index, key=lambda x: x['area'])


            for product in [p for p in self.sorted_products if p['quantity'] > 0]:
                    prod_size = product['size']
                    best_fit_stock_idx, best_fit_position, min_waste = None, None, float('inf')

                    # Scan stock in use to find stock with the least wasted space
                    for stock_data in self.inUsed_stocks:
                        stock_idx = stock_data.get('index')  
                        stock = stock_data['stock']  
                        position = self.find_best_position(
                            stock, 
                            width=prod_size[0], 
                            height=prod_size[1]  
                        )
                        if position is not None:
                            waste_area = self.calculate_waste(
                                stock, position[0], position[1], prod_size[0], prod_size[1]
                            )
                            if not (waste_area >= min_waste):
                                min_waste = waste_area
                                best_fit_stock_idx, best_fit_position = stock_idx, position 

                    # If not found in inUsed_stocks, select smallest stock from notUsed_stocks
                    if not best_fit_stock_idx is not None and not self.notUsed_stocks is None:
                        smallest_stock = self.notUsed_stocks.pop(0) 
                        self.inUsed_stocks.append(smallest_stock)  
                        stock_idx, stock = smallest_stock['index'], smallest_stock['stock']
                        position = self.find_best_position(stock, prod_size[0], prod_size[1])
                        if position:
                            best_fit_stock_idx = stock_idx
                            best_fit_position = position

                    # Place the product if you find a suitable location
                    if best_fit_stock_idx is not None:
                        return {
                            'stock_idx': best_fit_stock_idx,
                            "size": (prod_size[0],prod_size[1]),
                            'position': best_fit_position
                        }
            # Out of product
            return {'stock_idx': -1, 'size': (0, 0), 'position': (0, 0)}

    def find_best_position(self, stock, width, height):
        optimal_position = None
        minimum_waste = float('inf')
        num_rows, num_cols = self._get_stock_size_(stock)

        for row in range(num_rows - width + 1):
            for col in range(num_cols - height + 1):
                if self._can_place_(stock,(row, col), (width, height)) and not (row + width > num_rows) and not (col + height > num_cols):
                    current_waste = self.calculate_waste(stock, row, col, width, height)
                    if not (current_waste >= minimum_waste):
                        min_waste = current_waste
                        optimal_position = (row, col)
        return optimal_position            

    def safe_get_stock_size(self, stock):
        try:
            stock_size = self._get_stock_size_(stock)
            if isinstance(stock_size, (list, np.ndarray)) and len(stock_size) >= 2:
                try:
                    size_0 = float(stock_size[0]) if isinstance(stock_size[0], (int, float)) else 0
                    size_1 = float(stock_size[1]) if isinstance(stock_size[1], (int, float)) else 0
                    return [size_0, size_1]
                except (IndexError, TypeError):
                    return [0, 0]
            return [0, 0]
        except Exception as e:
            return [0, 0]


    def calculate_waste(self, stock, row, col, width, height):
        region = stock[row:row + height, col:col + width]
        empty_cells = np.count_nonzero(region == -1)
        used_area = width * height
        return empty_cells - used_area
