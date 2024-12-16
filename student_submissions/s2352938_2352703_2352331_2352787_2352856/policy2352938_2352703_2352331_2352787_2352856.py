from policy import Policy
import numpy as np
from scipy.optimize import linprog
from copy import deepcopy

class Policy2352938_2352703_2352331_2352787_2352856(Policy):    
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.policy_id = policy_id
        self.optimal_patterns = []
        self.isComputing = True
        self.drawing_counter = -1
        self.drawing_data = []
        self.list_stocks = []
        self.list_products = []
        self.keys = []
        self.sub_optimal_patterns = []

        if policy_id == 1:
            self.sub_optimal_patterns = []
        elif policy_id == 2:
            self.stock_buckets = {}
            self.bucket_size = 10 
            self.indices_prods = []
            self.sorted_prods = []

    def get_action(self, observation, info):
        if(self.isComputing):
            self.isComputing = False
            self.drawing_counter += 1
            self.implement_policy(observation,info)                
            return {
                "stock_idx": self.drawing_data[self.drawing_counter]["stock_idx"],
                "size": self.drawing_data[self.drawing_counter]["size"],
                "position": self.drawing_data[self.drawing_counter]["position"]
            }
        else:
            self.drawing_counter += 1
            if(self.drawing_counter == len(self.drawing_data)):
                self.drawing_counter = 0
                self.optimal_patterns.clear()
                self.drawing_data.clear()
                self.list_products.clear()
                self.list_stocks.clear()
                self.keys.clear()
                if self.policy_id == 1:
                    self.sub_optimal_patterns.clear()
                else:
                    self.stock_buckets.clear()
                    self.indices_prods.clear()
                    self.sorted_prods.clear()
                self.implement_policy(observation,info)
                return {
                    "stock_idx": self.drawing_data[self.drawing_counter]["stock_idx"],
                    "size": self.drawing_data[self.drawing_counter]["size"],
                    "position": self.drawing_data[self.drawing_counter]["position"]
                }
            else:
                return {
                    "stock_idx": self.drawing_data[self.drawing_counter]["stock_idx"],
                    "size": self.drawing_data[self.drawing_counter]["size"],
                    "position": self.drawing_data[self.drawing_counter]["position"]
                }

    def implement_policy(self,observation,info):
        initial_stocks = deepcopy(observation["stocks"])
        initial_prods = deepcopy(observation["products"])

        if self.policy_id == 1:
            self.furini_heuristic(initial_stocks,initial_prods)
            self.drawing_strips()
        else:
            self.lazy_init_heuristic(initial_stocks,initial_prods)
            self.drawing_patterns()

    def furini_heuristic(self, initial_stocks, initial_prods):
        # Student code here
        prod_num = 0
        for prod_idx,prod in enumerate(initial_prods):
            prod_info = {'id': str(prod_idx),"width": prod["size"][0], "height": prod["size"][1], "quantity": prod["quantity"]}
            self.list_products.append(prod_info)
            # prod_num += prod["quantity"]
            prod_info = {'id': str(prod_idx) + '-rotated',"width": prod["size"][1], "height": prod["size"][0], "quantity": prod["quantity"]}
            self.list_products.append(prod_info)
            prod_num += prod['quantity']
        self.list_products.sort(key=lambda x: (-x['height'], -x['width']))

        stock_id = 0
        for stock_i_idx,stock_i in enumerate(initial_stocks):
            stock_w, stock_h = self._get_stock_size_(stock_i)
            duplicated_stock_idx = -1
            for stock_idx,stock in enumerate(self.list_stocks):
                if min(stock_w,stock_h) == stock["width"] and max(stock_h,stock_w) == stock["length"]:
                    duplicated_stock_idx = stock_idx
                    break
            if duplicated_stock_idx != -1:
                self.list_stocks[duplicated_stock_idx]["quantity"] += 1
                self.list_stocks[duplicated_stock_idx]["stock_index"].append((stock_i_idx,stock_h > stock_w))
            else:
                stock_info = {'id': stock_id,"width": min(stock_w,stock_h), "length": max(stock_h,stock_w), "quantity": 1, "stock_index": [(stock_i_idx,stock_h > stock_w)], 'used': 0, 'rotated': stock_h > stock_w }
                stock_id += 1
                self.list_stocks.append(stock_info)
        stock_quantity = [stock['quantity'] for stock in self.list_stocks]
        self.list_stocks.sort(key=lambda x:x['width'] * x['length'])

        initial_patterns = []
        bin_counter = 0
        item_demand = {item['id']: item['quantity'] for item in self.list_products}

        for item in self.list_products:
            while item_demand[item['id']] > 0:
                bin_class_id = self.choose_appropriate_stock_type_for_prod(self.list_stocks,item)
                if bin_class_id == -1:
                    item_demand[item['id']] = 0
                    break
                bin_class = next(bc for bc in self.list_stocks if bc['id'] == bin_class_id)
    
                # Open a new bin of this class if possible
                bin_counter += 1
                current_bin = {'id': bin_counter, 'key': str(bin_class['id']),'bin_class_id': bin_class['id'], 'length': bin_class['length'], 'width': bin_class['width'], 'remaining_length': bin_class['length'], 'remaining_width': bin_class['width'], 'strips': []}
                initial_patterns.append(current_bin)

                while item_demand[item['id']] > 0 and current_bin['remaining_width'] >= item['height']:
                    # Initialize a new strip
                    strip_width = item['height']
                    strip_length = 0
                    strip_items = []

                    items_to_place = min(item_demand[item['id']],int(current_bin['length'] // item['width']))

                    if items_to_place == 0:
                        break  # Cannot place more items in this bin

                    strip_length = items_to_place * item['width']

                    if strip_width > current_bin['remaining_width']:
                        break  # Cannot place strip in remaining length

                    item_placement = {'item_class_id': item['id'], 'width': item['width'], 'height': item['height'],'quantity': items_to_place}
                    strip = {'length': strip_length, 'width': strip_width, 'items': [item_placement]}
                    current_bin['key'] += ''.join('_' + str(item['id']) for _ in range(items_to_place))
                    # Update bin and item demand
                    current_bin['remaining_width'] -= strip_width
                    item_demand[item['id']] -= items_to_place
                    if('-rotated' in item['id']):
                        item_demand[item['id'].replace('-rotated','')] -= items_to_place
                    else:
                        item_demand[item['id'] + '-rotated'] -= items_to_place
                    # Fill the strip with smaller items if possible (greedy procedure)
                    
                    strip_remaining_length = current_bin['length'] - strip['length']
                    if strip_remaining_length > 0:
                        for next_item in self.list_products:
                            if item_demand[next_item['id']] > 0 and next_item['width'] <= strip_remaining_length and next_item['height'] <= strip_width:
                                items_to_place = min(item_demand[next_item['id']],int(strip_remaining_length // next_item['width']))
                                if(items_to_place > 0):
                                    item_placement = {'item_class_id': next_item['id'], 'width': next_item['width'], 'height': next_item['height'],'quantity': items_to_place}
                                    current_bin['key'] += ''.join('_' + str(next_item['id']) for _ in range(items_to_place))
                                    strip['items'].append(item_placement)
                                    item_demand[next_item['id']] -= items_to_place
                                    if('-rotated' in next_item['id']):
                                        item_demand[next_item['id'].replace('-rotated','')] -= items_to_place
                                    else:
                                        item_demand[next_item['id'] + '-rotated'] -= items_to_place
                                    strip['length'] += items_to_place * next_item['width']
                                    strip_remaining_length -= items_to_place * next_item['width']
                                    if strip_remaining_length <= 0:
                                        break
                    current_bin['strips'].append(strip)
                
                while current_bin['remaining_width'] > 0:
                    canPlaceMore = False
                    for sub_item in self.list_products:
                        if item_demand[sub_item['id']] > 0 and current_bin['remaining_width'] >= sub_item['height']:
                            canPlaceMore = True

                            strip_width = sub_item['height']
                            strip_length = 0
                            strip_items = []

                            items_to_place = min(item_demand[sub_item['id']],int(current_bin['length'] // sub_item['width']))

                            if items_to_place == 0:
                                break  # Cannot place more items in this bin

                            # rows_needed = (items_to_place + max_items_in_row - 1) // max_items_in_row
                            strip_length = items_to_place * sub_item['width']

                            if strip_width > current_bin['remaining_width']:
                                break  # Cannot place strip in remaining length

                            item_placement = {'item_class_id': sub_item['id'], 'width': sub_item['width'], 'height': sub_item['height'],'quantity': items_to_place}
                            current_bin['key'] += ''.join('_' + str(sub_item['id']) for _ in range(items_to_place))

                            strip = {'length': strip_length, 'width': strip_width, 'items': [item_placement]}

                            # Update bin and item demand
                            current_bin['remaining_width'] -= strip_width
                            item_demand[sub_item['id']] -= items_to_place
                            if('-rotated' in sub_item['id']):
                                item_demand[sub_item['id'].replace('-rotated','')] -= items_to_place
                            else:
                                item_demand[sub_item['id'] + '-rotated'] -= items_to_place
                            
                            strip_remaining_length = current_bin['length'] - strip['length']
                            if strip_remaining_length > 0:
                                for next_item in self.list_products:
                                    if item_demand[next_item['id']] > 0 and next_item['width'] <= strip_remaining_length and next_item['height'] <= strip_width:
                                        items_to_place = min(item_demand[next_item['id']],int(strip_remaining_length // next_item['width']))
                                        if(items_to_place > 0):
                                            item_placement = {'item_class_id': next_item['id'],'width': next_item['width'], 'height': next_item['height'], 'quantity': items_to_place}
                                            current_bin['key'] += ''.join('_' + str(next_item['id']) for _ in range(items_to_place))
                                            strip['items'].append(item_placement)
                                            item_demand[next_item['id']] -= items_to_place
                                            if('-rotated' in next_item['id']):
                                                item_demand[next_item['id'].replace('-rotated','')] -= items_to_place
                                            else:
                                                item_demand[next_item['id'] + '-rotated'] -= items_to_place
                                            strip['length'] += items_to_place * next_item['width']
                                            strip_remaining_length -= items_to_place * next_item['width']
                                            if strip_remaining_length <= 0:
                                                break
                            current_bin['strips'].append(strip)
                    if canPlaceMore == False: break
        patterns_converted = []
        D = np.zeros(len(initial_prods))
        for prod in self.list_products:
            if '-rotated' in prod['id']: continue
            D[int(prod['id'])]=prod["quantity"]
        D = D.flatten()

        S = np.zeros(len(self.list_stocks))
        for stock in self.list_stocks:
            S[int(stock['id'])] = stock['quantity']
        S = S.flatten()
        
        c = np.array([])
        for pattern in initial_patterns:
            if pattern["key"] not in self.keys:
                self.keys.append(pattern["key"])
                unique_pattern = {'key': pattern['key'], "quantity": 1, "stock_type": pattern["bin_class_id"], "strips": pattern["strips"]}
                patterns_converted.append(unique_pattern)
                self.sub_optimal_patterns.append(unique_pattern)
                area = pattern['length'] * pattern['width']
                c = np.append(c,area)
            else:
                self.update_quantity_pattern_by_key(patterns_converted, pattern["key"])
        c = c.flatten()
        self.sub_optimal_patterns = deepcopy(patterns_converted)

        A = np.zeros(shape=(int(len(self.list_products) / 2),len(patterns_converted))) # 11 row - 28 col
        for pattern_idx, pattern in enumerate(patterns_converted):
            for strip in pattern['strips']:
                for item in strip['items']:
                    if '-rotated' in item['item_class_id']: item_idx = item['item_class_id'].replace('-rotated','')
                    else: item_idx = item['item_class_id']
                    item_idx = int(item_idx)
                    A[item_idx][pattern_idx] += item['quantity']

        B = np.zeros(shape=(len(self.list_stocks),len(patterns_converted))) # 97 row - 28 col
        for pattern_idx, pattern in enumerate(patterns_converted):
            B[pattern["stock_type"]][pattern_idx] = 1
        
        x_bounds = [(0,None) for _ in range(len(patterns_converted))]
        result_simplex = linprog(c,A_ub=B,b_ub=S,A_eq=A,b_eq=D,bounds=x_bounds,method='highs')
        if(result_simplex.status == 0):
            dual_prods = result_simplex.eqlin['marginals']
            dual_stocks = result_simplex.ineqlin['marginals']
            new_patterns_generation = []
            gen_status = True
            list_prod_sort_by_id = deepcopy(self.list_products)
            for prod in list_prod_sort_by_id:
                if '-rotated' in prod['id']:
                    prod['id'] = int(prod['id'].replace('-rotated', '')) * 2 + 1
                else:
                    prod['id'] = int(prod['id']) * 2
            list_prod_sort_by_id.sort(key=lambda x: x['id'], reverse=False)

            i = 0
            for prod in list_prod_sort_by_id:
                if prod['id'] % 2 != 0:
                    prod['id'] = str(i - 1) + '-rotated'
                else:
                    prod['id'] = str(i)
                    i += 1

            for i in range(len(self.list_stocks)): 
                new_strips = self.generate_pattern(dual_prods, i)
                if new_strips == None:
                    gen_status = False
                    break
                key = str(i)
                converted_strips = []
                for strip in new_strips:
                    items = []
                    length = 0
                    for item_count_idx,item_count in enumerate(strip['itemCount']):
                        if item_count == 0: continue
                        key += ''.join('_' + str(list_prod_sort_by_id[item_count_idx]['id']) for _ in range(item_count))
                        length += list_prod_sort_by_id[item_count_idx]['width'] * item_count
                        items.append({'item_class_id': list_prod_sort_by_id[item_count_idx]['id'], 'width': list_prod_sort_by_id[item_count_idx]['width'], 'height': list_prod_sort_by_id[item_count_idx]['height'], 'quantity': item_count})
                    converted_strips.append({'length': length, 'width': strip['strip'], 'items': items})
                new_patterns_generation.append({'key': key, 'quantity': 1, 'stock_type': i, 'strips': converted_strips})

            # Calculate reduce cost
            if gen_status:
                solveMilp = True
                reduce_costs = []
                for pattern in new_patterns_generation:
                    bin_class = next(bc for bc in self.list_stocks if bc['id'] == pattern['stock_type'])
                    new_column_A = np.zeros(int(len(self.list_products)/2))
                    for strip in pattern['strips']:
                        for item in strip['items']:
                            if '-rotated' in item['item_class_id']: item_idx = item['item_class_id'].replace('-rotated','')
                            else: item_idx = item['item_class_id']
                            item_idx = int(item_idx)
                            new_column_A[item_idx] += item['quantity']
                    reduce_cost = bin_class['width'] * bin_class['length'] - (np.dot(new_column_A,dual_prods.transpose())  + self.get_stock_price(dual_stocks,pattern['stock_type']))
                    if reduce_cost < 0 and pattern['key'] not in self.keys:
                        patterns_converted.append(pattern)
                        self.keys.append(pattern['key'])
                        solveMilp = False
                        c = np.append(c,bin_class['width'] * bin_class['length'])
                        A = np.column_stack((A, new_column_A))
                        new_column_B = np.zeros(int(len(self.list_stocks)))
                        new_column_B[pattern["stock_type"]] = 1
                        B = np.column_stack((B, new_column_B))

                if solveMilp:
                    self.solveMilp(D,S,c,A,B,patterns_converted,prod_num)
                else:
                    self.solveLp(D,S,c,A,B,patterns_converted,prod_num)
            else:
                self.optimal_patterns = self.sub_optimal_patterns
        else:
            self.optimal_patterns = self.sub_optimal_patterns

    def get_stock_price(self,dual_stocks,stock_type):
        for stock_idx, stock in enumerate(self.list_stocks):
            if stock['id'] == stock_type:
                return dual_stocks[stock_idx]

    def lazy_init_heuristic(self,initial_stocks,initial_prods):
        prod_num = 0
        for prod_idx,prod in enumerate(initial_prods):
            prod_info = {"width": prod["size"][0], "height": prod["size"][1], "quantity": prod["quantity"]}
            self.list_products.append(prod_info)
            prod_num += prod["quantity"]

        for stock_i_idx,stock_i in enumerate(initial_stocks):
            stock_w, stock_h = self._get_stock_size_(stock_i)
            duplicated_stock_idx = -1
            for stock_idx,stock in enumerate(self.list_stocks):
                if stock_w == stock["width"] and stock_h == stock["height"]:
                    duplicated_stock_idx = stock_idx
                    break
            if duplicated_stock_idx != -1:
                self.list_stocks[duplicated_stock_idx]["quantity"] += 1
                self.list_stocks[duplicated_stock_idx]["stock_index"].append(stock_i_idx)
            else:
                stock_info = {"width": stock_w, "height": stock_h, "quantity": 1, "stock_index": [stock_i_idx]}
                self.list_stocks.append(stock_info)

        # Pattern for all stocks
        # pattern = {'stock_idx': number, 'items': map[]}[]
        # map: (key-value) -> (prod_idx: {"quantity": number, "positions": number[][], "width": number, "height": number})
        
        # Initialize the pattern
        initial_patterns = []
        for stock_idx, stock in enumerate(initial_stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_type = -1
            for s_idx,s in enumerate(self.list_stocks):
                if s["width"] == stock_w and s["height"] == stock_h:
                    stock_type = s_idx
                    break
            pattern_element = {'key': '_' + str(stock_type) + '_','stock_idx': stock_idx, 'stock_type': stock_type, 'width': stock_w,'height': stock_h,'items': {}}
            initial_patterns.append(pattern_element)

        clone_stocks = initial_stocks
        clone_prods = initial_prods
        if len(self.indices_prods) == 0:
            self.indices_prods = list(range(len(clone_prods)))
        for _ in range(prod_num):
            heuristic_result = self.lazy_init(clone_prods, clone_stocks, self.indices_prods)
            prod_idx = heuristic_result["prod_idx"]
            best_stock_idx = heuristic_result["stock_idx"]
            best_position = heuristic_result["position"]
            best_prod_size = heuristic_result["size"]
            clone_stocks, clone_prods = self.fill_to_clone_stocks(clone_stocks, clone_prods, prod_idx, best_stock_idx, best_position, best_prod_size)
            if prod_idx in initial_patterns[best_stock_idx]["items"]:
                initial_patterns[best_stock_idx]["items"][prod_idx]["quantity"] += 1
                initial_patterns[best_stock_idx]["items"][prod_idx]["positions"].append(best_position)
                initial_patterns[best_stock_idx]["key"]+=str(prod_idx) + '_'
            else:
                position_list = [best_position]
                prod_w, prod_h = best_prod_size
                initial_patterns[best_stock_idx]["items"][prod_idx] = {"quantity": 1, "positions": position_list, "width": prod_w, "height": prod_h }
                initial_patterns[best_stock_idx]["key"]+=str(prod_idx) + '_'
        
        patterns_converted = []
        for pattern in initial_patterns:
            if pattern["items"] == {}: continue   
            if pattern["key"] not in self.keys:
                self.keys.append(pattern["key"])
                unique_pattern = {"key": pattern['key'], "quantity": 1, "stock_type": pattern["stock_type"], "items": pattern["items"]}
                patterns_converted.append(unique_pattern)
            else:
                self.update_quantity_pattern_by_key(patterns_converted,pattern['key'])
        self.optimal_patterns = patterns_converted
            
    def fill_to_clone_stocks(self,clone_stocks, clone_prods, prod_idx, best_stock_idx, best_position, best_prod_size):
        x, y = best_position
        w, h = best_prod_size
        for i in range(x, x + w):
            for j in range(y, y + h):
                clone_stocks[best_stock_idx][i][j] = prod_idx
        return clone_stocks, clone_prods

    def lazy_init(self, clone_prods, clone_stocks, indices_prods):
        best_stock_idx, best_position, best_prod_size = -1, None, [0, 0]
        if self.sorted_prods == []:
            self.sorted_prods = sorted(clone_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)
            self.indices_prods = sorted(self.indices_prods, key=lambda i: clone_prods[i]["size"][0] * clone_prods[i]["size"][1], reverse=True)

        clone_prods = self.sorted_prods
        # Group stocks into buckets based on size ranges
        self._group_stocks_into_buckets(clone_stocks)
        
        for prod_idx,prod in enumerate(clone_prods):
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                min_waste_percentage = float('inf')
                candidate_stocks = self._get_candidate_stocks(prod_size)
                
                for stock_idx, stock in candidate_stocks:
                    placed = False
                    position = self._find_position(stock, prod_size[0], prod_size[1])
                    if position:
                        stock_w, stock_h = self._get_stock_size_(stock)
                        stock_area = stock_w * stock_h
                        prod_area = prod_size[0] * prod_size[1]
                        waste_percentage = (stock_area - prod_area) / stock_area
                        if waste_percentage < min_waste_percentage:
                            min_waste_percentage = waste_percentage
                            best_stock_idx = stock_idx
                            best_position = position
                            best_prod_size = prod_size
                            placed = True
                            break
                if best_position and best_stock_idx != -1:
                    prod["quantity"] -= 1
                    return {"prod_idx": self.indices_prods[prod_idx], "stock_idx": best_stock_idx, "size": (best_prod_size[0], best_prod_size[1]), "position": best_position}
        return {"stock_idx": -1, "size": [0, 0], "position": None}

    def _group_stocks_into_buckets(self, stocks):
        self.stock_buckets = {}
        for idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            bucket_key = (stock_w // self.bucket_size, stock_h // self.bucket_size)
            if bucket_key not in self.stock_buckets:
                self.stock_buckets[bucket_key] = []
            self.stock_buckets[bucket_key].append((idx, stock))

    def _get_candidate_stocks(self, prod_size):
        prod_w, prod_h = prod_size
        bucket_key = (prod_w // self.bucket_size, prod_h // self.bucket_size)
        candidate_stocks = []
        for key in self.stock_buckets:
            if key[0] >= bucket_key[0] and key[1] >= bucket_key[1]:
                candidate_stocks.extend(self.stock_buckets[key])
        return candidate_stocks

    def _find_position(self, stock, product_width, product_height):
        stock_width, stock_height = self._get_stock_size_(stock)

        for x in range(stock_width - product_width + 1):
            for y in range(stock_height - product_height + 1):
                if self._can_place_(stock, (x, y), (product_width, product_height)):
                    return (x, y)
        return None

    #########################################################################

    def solveLp(self,D,S,c,A,B,patterns_converted,prod_num):
        x_bounds = [(0,None) for _ in range(len(patterns_converted))]
        result_simplex = linprog(c,A_ub=B,b_ub=S,A_eq=A,b_eq=D,bounds=x_bounds,method='highs')
        if(result_simplex.status == 0):
            dual_prods = result_simplex.eqlin['marginals']
            dual_stocks = result_simplex.ineqlin['marginals']
            new_patterns_generation = []
            
            list_prod_sort_by_id = deepcopy(self.list_products)
            for prod in list_prod_sort_by_id:
                if '-rotated' in prod['id']:
                    prod['id'] = int(prod['id'].replace('-rotated', '')) * 2 + 1
                else:
                    prod['id'] = int(prod['id']) * 2
            list_prod_sort_by_id.sort(key=lambda x: x['id'], reverse=False)

            i = 0
            for prod in list_prod_sort_by_id:
                if prod['id'] % 2 != 0:
                    prod['id'] = str(i - 1) + '-rotated'
                else:
                    prod['id'] = str(i)
                    i += 1

            gen_status = True
            for i in range(len(self.list_stocks)):
                new_strips = self.generate_pattern(dual_prods, i)
                if new_strips == None:
                    gen_status = False
                    break
                key = str(i)
                converted_strips = []
                for strip in new_strips:
                    items = []
                    length = 0
                    for item_count_idx,item_count in enumerate(strip['itemCount']):
                        if item_count == 0: continue
                        key += ''.join('_' + str(list_prod_sort_by_id[item_count_idx]['id']) for _ in range(item_count))
                        length += list_prod_sort_by_id[item_count_idx]['width'] * item_count
                        items.append({'item_class_id': list_prod_sort_by_id[item_count_idx]['id'], 'width': list_prod_sort_by_id[item_count_idx]['width'], 'height': list_prod_sort_by_id[item_count_idx]['height'], 'quantity': item_count})
                    converted_strips.append({'length': length, 'width': strip['strip'], 'items': items})
                new_patterns_generation.append({'key': key, 'quantity': 1, 'stock_type': i, 'strips': converted_strips})

            # Calculate reduce cost
            if gen_status:
                solveMilp = True
                reduce_costs = []
                for pattern in new_patterns_generation:
                    bin_class = next(bc for bc in self.list_stocks if bc['id'] == pattern['stock_type'])
                    new_column_A = np.zeros(int(len(self.list_products)/2))
                    for strip in pattern['strips']:
                        for item in strip['items']:
                            if '-rotated' in item['item_class_id']: item_idx = item['item_class_id'].replace('-rotated','')
                            else: item_idx = item['item_class_id']
                            item_idx = int(item_idx)
                            new_column_A[item_idx] += item['quantity']
                    reduce_cost = bin_class['width'] * bin_class['length'] - (np.dot(new_column_A,dual_prods.transpose())  + self.get_stock_price(dual_stocks,pattern['stock_type']))
                    if reduce_cost < 0 and pattern['key'] not in self.keys:
                        patterns_converted.append(pattern)
                        self.keys.append(pattern['key'])
                        solveMilp = False
                        c = np.append(c,bin_class['width'] * bin_class['length'])
                        A = np.column_stack((A, new_column_A))
                        new_column_B = np.zeros(int(len(self.list_stocks)))
                        new_column_B[pattern["stock_type"]] = 1
                        B = np.column_stack((B, new_column_B))

                if solveMilp:
                    self.solveMilp(D,S,c,A,B,patterns_converted,prod_num)
                else:
                    self.solveLp(D,S,c,A,B,patterns_converted,prod_num)
            else:
                self.optimal_patterns = self.sub_optimal_patterns
        else:
            self.optimal_patterns = self.sub_optimal_patterns

    def solveMilp(self,D,S,c,A,B,patterns_converted,prod_num):
        x_bounds = [(0,None) for _ in range(len(c))]
        optimal_result = linprog(c,A_ub=B,b_ub=S,A_eq=A,b_eq=D,bounds=x_bounds,method='highs',integrality=1,options={'presolve': False})
        if optimal_result.status == 0:
            patterns_quantity = np.int_(optimal_result.x)
            total_area = 0
            for pattern_idx,pattern in enumerate(patterns_converted):
                pattern['quantity'] = patterns_quantity[pattern_idx]
            prod_sum = 0
            for pattern in patterns_converted:
                if pattern['quantity'] == 0: continue
                prod_sum += pattern['key'].count('_') * pattern['quantity']
            if prod_sum != prod_num:
                self.optimal_patterns = self.sub_optimal_patterns
            else: 
                self.optimal_patterns = patterns_converted
        else:
            self.optimal_patterns = self.sub_optimal_patterns

    def choose_appropriate_stock_type_for_prod(self,list_stocks,item):
        max_items_in_bin = 0
        assigned_bin_class = None
        for bin_class in list_stocks:
            if bin_class['used'] < bin_class['quantity']:
                max_items = int((bin_class['width'] // item['height']) * (bin_class['length'] // item['width']))
                if max_items >= item['quantity']:
                    assigned_bin_class = bin_class
                    break
                elif max_items > max_items_in_bin:
                    max_items_in_bin = max_items
                    assigned_bin_class = bin_class
        if assigned_bin_class:
            assigned_bin_class['used'] += 1
            return assigned_bin_class['id']
        else:
            return -1

    def get_stock_by_type(self,stock_type):
        for stock in self.list_stocks:
            if stock['id'] == stock_type:
                return stock

    def drawing_strips(self):
        for data in self.optimal_patterns:
            if data['quantity'] == 0: continue
            for _ in range(data['quantity']):
                stock = self.get_stock_by_type(data['stock_type'])
                stock_idx_info = stock['stock_index'][0]
                stock['stock_index'].pop(0)
                x,y = 0,0
                for strip in data['strips']:
                    if stock_idx_info[1]:
                        for item in strip['items']:
                            for _ in range(item['quantity']):
                                size = (item['height'],item['width'])
                                position = (x,y)
                                y += item['width']
                                self.drawing_data.append({
                                    'stock_idx': stock_idx_info[0],
                                    'size': size,
                                    'position': position,
                                })
                        x += strip['width']
                        y = 0
                    else:
                        for item in strip['items']:
                            for _ in range (item['quantity']):
                                size = (item['width'],item['height'])
                                position = (x,y)
                                x += item['width']
                                self.drawing_data.append({
                                    'stock_idx': stock_idx_info[0],
                                    'size': size,
                                    'position': position,
                                })
                        y += strip['width']
                        x = 0

    def drawing_patterns(self):
        for data in self.optimal_patterns:
            if data['quantity'] == 0: continue
            for _ in range(data['quantity']):
                stock_type = data['stock_type']
                stock_idx = self.list_stocks[stock_type]['stock_index'][0]
                self.list_stocks[stock_type]['stock_index'].pop(0)
                items = data['items']
                if items:
                    for item_id, details in items.items():
                        size = (details['width'], details['height'])
                        positions = details['positions']
                        for position in positions:
                            self.drawing_data.append({
                                'stock_idx': stock_idx,
                                'size': size,
                                'position': position,
                            })
    
    def update_quantity_pattern_by_key(self,patterns_converted,key):
        for pattern in patterns_converted:
            if(pattern['key'] == key):
                pattern['quantity'] += 1
                break

    def generate_pattern(self, dual_prods, stock_type):
        #Initialize
        top = 6
        result = []
        stock = self.get_stock_by_type(stock_type)
        stock_w = stock['length']
        stock_h = stock['width']

        clone_products = deepcopy(self.list_products)
        clone_products.sort(key=lambda x:x['id'])
        for prod in clone_products:
            if '-rotated' in prod['id']:
                prod['id'] = int(prod['id'].replace('-rotated', '')) * 2 + 1
            else:
                prod['id'] = int(prod['id']) * 2

        clone_products.sort(key=lambda x: x['id'])
        product_widths = np.array([prod['width'] for prod in clone_products])
        product_heights = np.array([prod['height'] for prod in clone_products])
        product_quantity = np.array([prod['quantity'] for prod in clone_products])

        # cut horizontal strips
        # get all possible strip can be form
        strip_height_unique = set()
        for i in range(len(product_heights)):
            if product_widths[i] > stock_w or product_heights[i] > stock_h:
                continue
            if product_heights[i] in strip_height_unique: continue
            else: strip_height_unique.add(product_heights[i])
            profit = np.zeros(stock_w + 1)
            itemCount = np.zeros(len(self.list_products))
            for j in range(len(product_widths)):
                if product_heights[j] > product_heights[i]:
                    continue
                itemCount[j] = min(clone_products[j]['quantity'],stock_w / product_widths[j])
            result.append({"strip": int(product_heights[i]), "profit": int(profit[stock_w]), "itemCount": itemCount.astype(int)})
        result = np.array(result)

        small_result = []

        prod_clone = deepcopy(self.list_products)
        for prod in prod_clone:
            if '-rotated' in prod['id']:
                prod['id'] = int(prod['id'].replace('-rotated', '')) * 2 + 1
            else:
                prod['id'] = int(prod['id']) * 2
        prod_clone.sort(key=lambda x: x['id'], reverse=False)
    
        top_strips = np.zeros((top * len(product_heights), 1), dtype=float)
        for strips in result:
            strips['profit'] = int(strips['profit'])
            strips['strip'] = int(strips['strip'])
            existing_patterns = set()
            for i in range(len(strips["itemCount"])):
                array_cal = np.zeros(len(strips["itemCount"]), dtype=int)
                array_cal[i] = int(strips["itemCount"][i])
                if clone_products[i]['height'] > strips["strip"]:
                    continue
                small_profit = array_cal[i] * dual_prods[int(clone_products[i]['id'] / 2)]
                small_l = product_widths[i] * array_cal[i]

                if small_l > stock_w:
                    continue

                # Add initial pattern
                pattern_key = tuple(array_cal)
                if pattern_key not in existing_patterns:
                    small_result.append({
                        "strip": int(strips["strip"]),
                        "profit": int(small_profit),
                        "itemCount": array_cal.copy()
                    })
                    existing_patterns.add(pattern_key)
                
                clone_array_cal = deepcopy(array_cal)
                prod_clone_clone = deepcopy(prod_clone)
                prod_clone_clone.sort(key=lambda x: dual_prods[int(x['id'] / 2)], reverse=False)
                for k in range(array_cal[i]-1, 0, -1):
                    greedy_profit = small_profit - dual_prods[int(prod_clone[i]['id'] / 2)] * (array_cal[i] - k)
                    clone_array_cal[i] = k
                    can_fit_w = stock_w - (clone_array_cal[i] * prod_clone[i]['width'])
                    for prod in prod_clone_clone:
                        if int(prod['id']) == i: continue
                        can_fit = can_fit_w // prod['width']
                        if (can_fit > 0) and prod['height'] <= strips['strip'] and prod['width'] <= stock_w:
                            clone_array_cal[int(prod['id'])] = can_fit
                            can_fit_w -= (can_fit * prod['width'])
                            greedy_profit += can_fit * dual_prods[int(prod['id'] / 2)]
                    small_result.append({
                        "strip": int(strips["strip"]),
                        "profit": int(greedy_profit),
                        "itemCount": clone_array_cal.copy()
                    })

                prod_clone.sort(key=lambda x: dual_prods[int(x['id'] / 2)], reverse=False)

                for j in range(len(prod_clone)):
                    if prod_clone[j]["height"] > strips["strip"] or prod_clone[j]["width"] > stock_w - small_l or prod_clone[j]['id'] == prod_clone[i]['id']:
                        continue
                    max_k = min(prod_clone[j]["quantity"], int((stock_w - small_l) // product_widths[j]))
                    for k in range(0, max_k + 1):
                        array_cal[prod_clone[j]["id"]] += 1
                        small_profit += dual_prods[int(prod_clone[j]["id"] / 2)]
                        small_l += product_widths[j]
                        if small_l > stock_w:
                            break
                        pattern_key = tuple(array_cal)
                        if pattern_key not in existing_patterns:
                            small_result.append({
                                "strip": int(strips["strip"]),
                                "profit": int(small_profit),
                                "itemCount": array_cal.copy()
                            })
                            existing_patterns.add(pattern_key)

                    # Reset for next iteration
                    small_profit -= dual_prods[int(prod_clone[j]["id"] / 2)] * max_k
                    small_l -= product_widths[j] * max_k
                    array_cal[prod_clone[j]["id"]] -= max_k
                prod_clone = deepcopy(prod_clone_clone)

        valid_patterns = []
        for strip in small_result:
            strip_length = 0
            for item_idx,item in enumerate(strip['itemCount']):
                strip_length += item * product_widths[item_idx]
            if strip_length <= stock_w:
                valid_patterns.append(strip)

        prod_heights = [prod['height'] for prod in prod_clone]
        valid_patterns.sort(key=lambda x: (x["strip"], -x["profit"]))

        h_strips = np.zeros((top * len(product_heights), 1), dtype = int)
        h_stock = np.zeros((top * len(product_heights), 1), dtype = int)
        min_array = np.zeros((top * len(product_heights), 1), dtype = int)
        strip_list = []

        strip_idx = [strip['strip'] for strip in valid_patterns]
        for i in range(len(prod_heights)):
            current_strips = [s for s in valid_patterns if s['strip'] == prod_heights[i]]
            seen_patterns = set()
            unique_strips = []
            for strip in current_strips:
                key = (strip['strip'], tuple(strip['itemCount']))
                if key not in seen_patterns:
                    seen_patterns.add(key)
                    unique_strips.append(strip)
            current_strips = unique_strips
            current_strips.sort(key=lambda x: -x['profit'])
            count = 0          
            for strip in current_strips[:top]:
                min_for_array = 1000000
                min2 = 1000000
                for j in range(len(strip['itemCount'])):
                    if strip['itemCount'][j] == 0: continue
                    d_i = min(int(self.list_products[j]['quantity']), int(stock_h * stock_w / (product_widths[j] * product_heights[j])))
                    min2 = min(d_i, min2, strip['itemCount'][j])
                min_for_array = min(min2, stock_h // strip['strip'])
                min_array[i*top+count][0] = min_for_array
                h_stock[i*top+count][0] = stock_h
                top_strips[i*top+count][0] = strip['profit']
                h_strips[i*top+count][0] = strip['strip']
                strip_list.append(strip)
                count += 1

        top_strips_unique = np.zeros(len(strip_list))
        h_strips_unique = np.zeros(len(strip_list))
        h_stock_unique = np.zeros(len(strip_list))
        min_array_unique = np.zeros(len(strip_list))
        for i in range(len(strip_list)):
            top_strips_unique[i] = strip_list[i]['profit']
            h_strips_unique[i] = strip_list[i]['strip']
            h_stock_unique[i] = stock_h
            min_array_unique[i] = stock_h // strip_list[i]['strip']

        ############################################################################################################
        prod_demand_constraints = np.array([])
        for product in clone_products:
            if product['id'] % 2 == 1: continue
            prod_demand_constraints = np.append(prod_demand_constraints,product['quantity'])

        prod_quantity_in_strip = np.zeros((int(len(clone_products)/2),len(strip_list)))
        for i in range(int(len(clone_products)/2)):
            for j in range(len(strip_list)):
                prod_quantity_in_strip[i][j] = strip_list[j]['itemCount'][i * 2] + strip_list[j]['itemCount'][i * 2 + 1]

        A = np.array([h_strips_unique.transpose()]).reshape(-1, h_strips_unique.shape[0])
        for i in range(0,len(min_array_unique)):
            constraint_binary = np.zeros(len(min_array_unique))
            constraint_binary[i] = 1
            A = np.vstack((A,constraint_binary))
        for quantity_strip in prod_quantity_in_strip:
            A = np.vstack((A,quantity_strip))

        b_u = np.array([stock_h])
        for min_array_element in min_array_unique.transpose():
            b_u = np.append(b_u,min_array_element)
        for demand_constraint in prod_demand_constraints:
            b_u = np.append(b_u,demand_constraint)
        
        b_l = np.full_like(b_u,-np.inf,dtype=float)

        c = -top_strips_unique.transpose().reshape(-1)
        res = linprog(c=c,A_ub=A,b_ub=b_u,bounds=(0,None),method='highs',integrality=1,options={'presolve': False})
        if(res.status != 0):
            return None
        else:
            return_res = []
            for i in range(len(res.x)):
                if res.x[i] > 0:
                    for j in range(int(res.x[i])):
                        return_res.append(strip_list[i])
            return return_res