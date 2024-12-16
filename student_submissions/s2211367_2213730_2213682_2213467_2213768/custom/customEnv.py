import gymnasium as gym
import numpy as np
import random
from gym_cutting_stock.envs.cutting_stock import CuttingStockEnv

class CustomCuttingStockEnv(CuttingStockEnv):
    def reset(self, seed=None, options=None):
        if(options == None ):
            return super().reset(seed=seed)    
        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        if options and 'stocks' in options:
            self._stocks = []
            for stock_data in options['stocks']:
                width, height = stock_data['size']
                stock = np.full(shape=(self.max_w, self.max_h), fill_value=-2, dtype=int)
                stock[:width, :height] = -1  # Empty cells are marked as -1
                self._stocks.append(stock)
            random.shuffle(self._stocks)    
            self._stocks = tuple(self._stocks)
            self.num_stocks = len(self._stocks)
        
        if options and 'products' in options:
            self._products = []
            for product_data in options['products']:
                width, height = np.array(product_data['size'])
                quantity = product_data['quantity']
                product = {"size": np.array([width, height]), "quantity": quantity}
                self._products.append(product)
            self._products = tuple(self._products)
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

def read_testcase_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    stock_sizes = lines[0].split(',')
    stock_sizes = [size.strip() for size in stock_sizes]
    stock_quantities = lines[1].split(',')
    stock_quantities = [int(qty.strip()) for qty in stock_quantities]

    stocks = []
    for size_str, qty in zip(stock_sizes, stock_quantities):
        width, height = map(int, size_str.split('x'))
        stock = {'size': [width, height]}
        stocks.extend([stock] * qty)

    products = []
    for line in lines[2:]:
        size_qty = line.split(',')
        size_str = size_qty[0].strip()
        quantity = int(size_qty[1].strip())
        width, height = map(int, size_str.split('x'))
        product = {'size': [width, height], 'quantity': quantity}
        products.append(product)

    return {'stocks': stocks, 'products': products}
