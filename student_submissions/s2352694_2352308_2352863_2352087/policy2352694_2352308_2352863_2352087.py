from policy import Policy
import numpy as np
import random

class Policy2352694_2352308_2352863_2352087(Policy):

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
            return False
        for x in range(pos_x, pos_x + prod_w):
            for y in range(pos_y, pos_y + prod_h):
                if stock[x, y] != -1:
                    return False
        return True

    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.stock_free_areas = {}

    def get_action(self, observation, info):
        if self.policy_id == 1:
            action = self.first_fit_descending(observation, info)
        elif self.policy_id == 2:
            if not self.stock_free_areas:
                self.stock_free_areas = {
                    idx: [ (0, 0, *self._get_stock_size_(stock)) ]
                    for idx, stock in enumerate(observation['stocks'])
                }
            action = self.best_fit_decreasing(observation, info)
        return action if action else {'stock_idx': 0, 'size': np.array([0, 0]), 'position': np.array([0, 0])}

    def first_fit_descending(self, observation, info):
        stocks = observation['stocks']
        products = observation['products']
        available_products = [
            {'id': idx, 'size': prod['size'], 'quantity': prod['quantity']}
            for idx, prod in enumerate(products) if prod['quantity'] > 0
        ]
        available_products.sort(key=lambda x: x['size'][0] * x['size'][1], reverse=True)
        stock_sizes = [
            (idx, np.sum(np.any(stock != -2, axis=1)) * np.sum(np.any(stock != -2, axis=0)))
            for idx, stock in enumerate(stocks)
        ]
        sorted_stocks = sorted(stock_sizes, key=lambda x: x[1], reverse=True)
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        for product in available_products:
            piece_size = product['size']
            for stock_idx, stock_size in sorted_stocks:
                stock = stocks[stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                if stock_w >= piece_size[0] and stock_h >= piece_size[1]:
                    pos_x, pos_y = None, None
                    for x in range(stock_w - piece_size[0] + 1):
                        for y in range(stock_h - piece_size[1] + 1):
                            if self._can_place_(stock, (x, y), piece_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        return {'stock_idx': stock_idx, 'size': piece_size, 'position': (pos_x, pos_y)}
                if stock_w >= piece_size[1] and stock_h >= piece_size[0]:
                    pos_x, pos_y = None, None
                    for x in range(stock_w - piece_size[1] + 1):
                        for y in range(stock_h - piece_size[0] + 1):
                            if self._can_place_(stock, (x, y), piece_size[::-1]):
                                piece_size = piece_size[::-1]
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        return {'stock_idx': stock_idx, 'size': piece_size, 'position': (pos_x, pos_y)}
        return None

    def best_fit_decreasing(self, observation, info):
        stocks = observation['stocks']
        products = observation['products']
        available_products = [
            {'id': idx, 'size': prod['size'], 'quantity': prod['quantity']}
            for idx, prod in enumerate(products) if prod['quantity'] > 0
        ]
        available_products.sort(key=lambda x: x['size'][0] * x['size'][1], reverse=True)
        for product in available_products:
            piece_size = product['size']
            piece_quantity = product['quantity']
            while piece_quantity > 0:
                min_waste = float('inf')
                best_placement = None
                for stock_idx, free_areas in self.stock_free_areas.items():
                    stock = stocks[stock_idx]
                    for area in free_areas:
                        x, y, w, h = area
                        for rotated in [False, True]:
                            pw, ph = piece_size if not rotated else piece_size[::-1]
                            if pw <= w and ph <= h:
                                position = (x, y)
                                if self._can_place_(stock, position, (pw, ph)):
                                    waste = (w * h) - (pw * ph)
                                    if waste < min_waste:
                                        min_waste = waste
                                        best_placement = {
                                            'stock_idx': stock_idx,
                                            'position': position,
                                            'size': (pw, ph),
                                            'rotated': rotated,
                                            'area': area
                                        }
                if best_placement:
                    free_areas = self.stock_free_areas[best_placement['stock_idx']]
                    free_areas.remove(best_placement['area'])
                    x, y = best_placement['position']
                    pw, ph = best_placement['size']
                    area_x, area_y, area_w, area_h = best_placement['area']
                    new_areas = []
                    if y > area_y:
                        new_areas.append((area_x, area_y, area_w, y - area_y))
                    if y + ph < area_y + area_h:
                        new_areas.append((area_x, y + ph, area_w, area_y + area_h - (y + ph)))
                    if x > area_x:
                        new_areas.append((area_x, y, x - area_x, ph))
                    if x + pw < area_x + area_w:
                        new_areas.append((x + pw, y, area_x + area_w - (x + pw), ph))
                    free_areas.extend([
                        rect for rect in new_areas if rect[2] > 0 and rect[3] > 0
                    ])
                    piece_quantity -= 1
                    return {
                        'stock_idx': best_placement['stock_idx'],
                        'size': np.array(piece_size if not best_placement['rotated'] else piece_size[::-1]),
                        'position': np.array(best_placement['position'])
                    }
                else:
                    break
        return None

    def _find_position_(self, stock, piece_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        for x in range(stock.shape[0] - piece_size[0] + 1):
            for y in range(stock.shape[1] - piece_size[1] + 1):
                if self._can_place_(stock, (x, y), piece_size):
                    return (x, y)
        return None

    def _place_piece_(self, stock, position, piece_size):
        x, y = position
        w, h = piece_size
        stock[x:x+w, y:y+h] = -3

    def _calculate_trim_loss_(self, original_stock, used_stock):
        total_area = np.sum(original_stock != -2)
        used_area = np.sum(used_stock == -3)
        return total_area - used_area

    def _solution_to_action_(self, solution, stocks):
        for placement in solution:
            stock_idx = placement['stock_idx']
            position = placement['position']
            size = placement['size']
            stock = stocks[stock_idx]
            if self._can_place_(stock, position, size):
                return {
                    'stock_idx': stock_idx,
                    'size': np.array(size),
                    'position': np.array(position)
                }
        return None
