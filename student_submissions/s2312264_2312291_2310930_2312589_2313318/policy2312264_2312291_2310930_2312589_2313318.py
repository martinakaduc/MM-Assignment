from policy import Policy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # input_shape = (1, 1, 100, 100)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding = 2),  # (1, 32, 51, 51)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Pooling layer
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding = 2),  # (1, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer   (1, 64, 7, 7)
        )
        

        self.flatten_cnn = nn.Flatten()


        self.fc_cnn = nn.Linear(32* 7 * 7, 200)  
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(75, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        
        self.fc_combined = nn.Sequential(
            nn.Linear(300, 100),  
            nn.ReLU(),
            nn.Linear(100, 25),  
        )
    
    def forward(self, x_cnn, x_mlp):
        x_cnn = self.cnn_layers(x_cnn)
        x_cnn = self.flatten_cnn(x_cnn)
        x_cnn = F.relu(self.fc_cnn(x_cnn))
        
        x_mlp = self.mlp_layers(x_mlp)
        
        x_combined = torch.cat((x_cnn, x_mlp), dim=1)
        output = self.fc_combined(x_combined)
        
        return output
        
class Policy2312264_2312291_2310930_2312589_2313318(Policy):
    def __init__(self, policy_id=1):
        super().__init__()
        assert policy_id in [1, 2, 3], "Policy ID must be 1 or 2"
        
        if policy_id == 1:
            self.counter = 0  # Initial counter = 0 for the times was called
            self.details = []  # Initial actions details
            self.pID = 1
        elif policy_id == 2:
            self.pID = 2
            self.cnt            = 0
            self.stock          = np.array([])
            self.step           = 0
            self.curIdx         = 0
            self.overfit        = 0
            self.GAMMA          = 0.99
            self.net            = Net()
            self.optim          = torch.optim.Adam(self.net.parameters(), lr = 0.0001)
        
            abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
            checkpoint = torch.load(abs_path, weights_only=True)
            #self.net.load_state_dict(torch.load('model.pth', weights_only=True))
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            # Student code here
        elif policy_id == 3:
            self.pID = 3
            self.free_rectangles = {}
            self.previous_stock_states = None
        
    
    def pick_sample(self, X, Y, stock, prods):
        product_idx = -1
        X_batch = torch.tensor(X, dtype=torch.float)
        Y_batch = (Y - np.min(Y, axis = -1))/ (np.max(Y, axis = -1) - np.min(Y, axis = -1) + 1e-8)
        Y_batch = torch.tensor(Y_batch, dtype=torch.float)
        
        logits = self.net.forward(X_batch, Y_batch)
        logits = torch.squeeze(logits, dim = 0)
        
        probs = F.softmax(logits, dim = -1)
        product_idx = torch.multinomial(probs, num_samples=1)
        product_idx = int(product_idx)
        prod_w, prod_h, quan = Y[0][3 * (product_idx + 1) - 3], Y[0][3 *(product_idx + 1) - 2], Y[0][3 *(product_idx + 1) - 1]
        if (Y[0][3 * (product_idx + 1) - 1] == 0):
            r = -1
        elif self.canPlace(stock, (prod_w, prod_h)) == False:
            r = -1
        else:
            r = 1
        
        return (product_idx, torch.log(probs[int(product_idx)]), r)
    
    def canPlace(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        if (prod_w == 0 or prod_h == 0):
            return False
        
        flag = False
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    flag = True
                    break
                if flag == True:
                    break

            if flag == True:
                break
        if flag == False:
            for x in range(stock_w - prod_h + 1):
                for y in range(stock_h - prod_w + 1):
                    if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                        flag = True
                        break
                    if flag == True:
                        break

                if flag == True:
                    break

        return flag == True
    # return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    def get_action(self, observation, info):
        if (self.pID == 1):
            if not self.details:  
                products = self.extract_products(observation["products"]) 
                stocks = self.extract_stocks(observation["stocks"]) 
                num, sheets, self.details, waste = self.cutting_stock_2d_min_waste(stocks, products)
                self.details = self.optimize_sheets(self.details, stocks) 
                self.details = self.last_sheet(self.details, stocks) 
            
            # If be called, counter += 1
            self.counter += 1
            
            # Return action until counter > len(self.details)
            if self.counter - 1 < len(self.details):
                return self.details[self.counter - 1]
            else:
                return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
        elif (self.pID == 2):
            stocks = np.array(observation['stocks'])  # ([])
            prods  = np.array(observation['products']) # ({})
            #X.shape = (1, 100, 100)
            
            
            if (np.size(self.stock) == 0 or (np.sum(stocks[self.curIdx] >= 0) == 0 and np.sum(self.stock == -2) != np.sum(stocks[self.curIdx] == -2))):
                self.step += 1
                self.stock = np.array(stocks[self.curIdx])
            if (self.overfit >= 10):
                self.step += 1
                self.curIdx += 1
                self.overfit = 0
                if (self.curIdx == 100):
                    self.curIdx = 0
                self.stock = np.array(stocks[self.curIdx])
            

            
            self.stock[self.stock >= 0] = -2
            X = np.array(self.stock)

            X = np.expand_dims(X, axis = 0)
            X = np.expand_dims(X, axis = 0)
            
            
            triplets = []
            for item in prods:
                prod_w, prod_h = item['size']
                q = item['quantity']
                triplets.append((prod_w, prod_h, q))
            while len(triplets) < 25:
                triplets.append((0, 0, 0))

            Y = np.array(triplets).flatten()
            Y = np.expand_dims(Y, axis=0)
            
            
            prod_idx, log_prob, r = self.pick_sample(X, Y, self.stock, prods)
            
            pos_x = 0
            pos_y = 0
            prod_w, prod_h = 0, 0
            stock_w, stock_h = self._get_stock_size_(self.stock)
            if (r == 1):
                self.overfit = 0
                prod_w, prod_h = prods[prod_idx]['size']
                flag = False
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(self.stock, (x, y),(prod_w, prod_h)):
                            flag = True
                            pos_x, pos_y = x, y
                            break
                        if flag == True:
                            break

                    if flag == True:
                        break
                if flag == False:
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(self.stock, (x, y), (prod_h, prod_w)):
                                flag = True
                                prod_w, prod_h = prod_h, prod_w
                                pos_x, pos_y = x, y
                                break
                            if flag == True:
                                break

                        if flag == True:
                            break
                if flag == True:
                    self.stock[pos_x : pos_x + prod_w, pos_y :  pos_y + prod_h] = -2
            else:
                self.overfit += 1
            
            return ({"stock_idx": self.curIdx, "size": (prod_w, prod_h), "position": (pos_x, pos_y)})
        elif (self.pID == 3):
            stocks = observation["stocks"]

            if self._is_new_episode(stocks):
                self.free_rectangles = {}
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    self.free_rectangles[stock_idx] = [{
                        "x": 0, "y": 0, "w": stock_w, "h": stock_h
                    }]
                self.previous_stock_states = [stock.copy() for stock in stocks]

            products = observation["products"]
            available_products = [prod for prod in products if prod["quantity"] > 0]

            if not available_products:
                return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

            available_products.sort(
                key=lambda p: p["size"][0] * p["size"][1], reverse=True
            )

            best_placement = None
            min_waste = float('inf')

            for prod in available_products:
                for test_w, test_h in [(prod["size"][0], prod["size"][1]),
                                    (prod["size"][1], prod["size"][0])]:
                    
                    for stock_idx, stock in enumerate(stocks):
                        free_rects = self.free_rectangles[stock_idx]

                        for rect in free_rects:
                            if test_w <= rect["w"] and test_h <= rect["h"]:
                                waste = rect["w"] * rect["h"] - test_w * test_h
                                if waste < min_waste:
                                    min_waste = waste
                                    best_placement = {
                                        "stock_idx": stock_idx,
                                        "position": (rect["x"], rect["y"]),
                                        "size": (test_w, test_h)
                                    }
                                    if waste == 0:  
                                        break
                        if min_waste == 0:
                            break
                    if min_waste == 0:
                        break
                if min_waste == 0:
                    break

            if best_placement is not None:
                stock_idx = best_placement["stock_idx"]
                pos_x, pos_y = best_placement["position"]
                prod_w, prod_h = best_placement["size"]

                self._split_free_rectangle(stock_idx, pos_x, pos_y, prod_w, prod_h)

                return {
                    "stock_idx": stock_idx,
                    "size": best_placement["size"],
                    "position": best_placement["position"]
                }

            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def extract_stocks(self, materials):
        stocks = []
        for material in materials:
            stock_w = int(np.sum(np.any(material != -2, axis=1)))
            stock_h = int(np.sum(np.any(material != -2, axis=0)))
            quantity = 1  # Default quantity is 1
            stocks.append((stock_w, stock_h, quantity))
        return stocks
    
    def extract_products(self, items):
        products = []
        for item in items:
            size = item['size']
            quantity = item['quantity']
            products.append((size[0], size[1], quantity))
        return products
    
    def cutting_stock_2d_min_waste(self, stocks, products):
        products = [(w, h, q) for w, h, q in products]
        # Sort products descending
        products.sort(key=lambda x: (x[0] * x[1]), reverse=True)

        sheets = []
        # Stock usage for check stock was used or not (default: 0 is not)
        stock_usage = {(i, w, h, q): 0 for i, (w, h, q) in enumerate(stocks)}

        def place_products(sheet_width, sheet_height, products):
            sheet_placements = []
            remaining_products = []
            placement_details = []
            used_area = 0

            # Arrange products and calculate waste
            for iw, ih, count in products:
                placed_count = 0
                for _ in range(count):
                    placed = False
                    for (cw, ch) in [(iw, ih), (ih, iw)]: # Both rotate and not
                        for y in range(sheet_height):
                            for x in range(sheet_width):
                                # Check if can be placed
                                if x + cw <= sheet_width and y + ch <= sheet_height and not overlap(
                                    sheet_placements, x, y, cw, ch):
                                    sheet_placements.append((x, y, cw, ch))
                                    placement_details.append({
                                        "stock_idx": None, 
                                        "size": (int(cw), int(ch)),
                                        "position": (x, y)
                                    })
                                    used_area += cw * ch
                                    placed = True
                                    placed_count += 1
                                    break
                            if placed:
                                break
                        if placed:
                            break
                if placed_count < count:
                    remaining_products.append((iw, ih, count - placed_count))

            total_area = sheet_width * sheet_height
            waste_area = total_area - used_area
            return sheet_placements, remaining_products, placement_details, waste_area

        def overlap(placements, x, y, iw, ih):
            for px, py, pw, ph in placements:
                if not (x + iw <= px or x >= px + pw or y + ih <= py or y >= py + ph):
                    return True
            return False

        # Add index
        sorted_stocks = [(i, w, h, q) for i, (w, h, q) in enumerate(stocks)]
        # Sort stocks descending
        sorted_stocks.sort(key=lambda x: (x[1] * x[2]), reverse=True)

        remaining_products = products[:]
        all_placement_details = []
        total_waste_area = 0

        while remaining_products:
            best_placements = []
            best_sheet = None
            best_remaining_products = []
            best_details = []
            min_waste_area = float('inf')

            for idx, (orig_idx, width, height, max_quantity) in enumerate(sorted_stocks):
                if stock_usage[(orig_idx, width, height, max_quantity)] < 1:
                    placements_on_sheet, remaining_products_on_sheet, details, waste_area = place_products(width, height, remaining_products)
                    # Look for the best sheet with min waste
                    if placements_on_sheet and waste_area < min_waste_area:
                        best_placements = placements_on_sheet
                        best_sheet = (width, height)
                        best_remaining_products = remaining_products_on_sheet
                        best_details = details
                        min_waste_area = waste_area
                        for detail in best_details:
                            detail["stock_idx"] = orig_idx # Add original index
                        break

            if best_placements:
                sheets.append((best_sheet, best_placements))
                stock_usage[(orig_idx, best_sheet[0], best_sheet[1], max_quantity)] += 1 # Mark as used
                remaining_products = best_remaining_products
                all_placement_details.extend(best_details)
                total_waste_area += min_waste_area
            else:
                return len(sheets), sheets, all_placement_details, total_waste_area
            
        # Return num sheets used, sheets, details of sheets and waste area
        return len(sheets), sheets, all_placement_details, total_waste_area
        
    # Optimize to use smaller stocks for min waste
    def optimize_sheets(self, all_placement_details, stocks):
        # Sort stocks descending
        sorted_stocks = [(i, w, h, q) for i, (w, h, q) in enumerate(stocks)]
        sorted_stocks.sort(key=lambda x: (x[1] * x[2]), reverse=True)
        # Add index
        stocks = [(i, w, h, q) for i, (w, h, q) in enumerate(stocks)]
        
        # Check the bounding area of all products can fit a stock
        def can_fit_in_stock(placements, stock_width, stock_height):
            if not placements:
                return False

            min_x = min(p["position"][0] for p in placements)
            max_x = max(p["position"][0] + p["size"][0] for p in placements)
            min_y = min(p["position"][1] for p in placements)
            max_y = max(p["position"][1] + p["size"][1] for p in placements)
            bounding_width = max_x - min_x
            bounding_height = max_y - min_y

            return bounding_width <= stock_width and bounding_height <= stock_height

        # Group stock details by index
        stock_details_by_idx = {}
        for detail in all_placement_details:
            stock_idx = detail["stock_idx"]
            if stock_idx not in stock_details_by_idx:
                stock_details_by_idx[stock_idx] = []
            stock_details_by_idx[stock_idx].append(detail)

        for current_stock_idx, placements_in_current_stock in stock_details_by_idx.items():
            current_stock = stocks[current_stock_idx]
            
            # Find smaller stocks and was not used (available_smaller_stocks)
            smaller_stocks = [
                (idx, w, h, q) for (idx, w, h, q) in sorted_stocks
                if idx != current_stock_idx and w * h < current_stock[1] * current_stock[2]
            ]
            
            available_smaller_stocks = [
                (new_stock_idx, new_width, new_height, new_quantity) for new_stock_idx, new_width, new_height, new_quantity in smaller_stocks
                if not any(detail["stock_idx"] == new_stock_idx for detail in all_placement_details)
            ]

            for new_stock_idx, new_width, new_height, new_quantity in available_smaller_stocks:
                if can_fit_in_stock(placements_in_current_stock, new_width, new_height):
                    for p in placements_in_current_stock:
                        # Update stock_idx
                        p["stock_idx"] = new_stock_idx  
            
        return all_placement_details
    
    # Find the best last sheet for min waste
    def last_sheet(self, all_placement_details, stocks):
        # Add index
        stocks = [(i, w, h, q) for i, (w, h, q) in enumerate(stocks)]

        # Group stock details by index
        stock_details_by_idx = {}
        for detail in all_placement_details:
            stock_idx = detail["stock_idx"]
            if stock_idx not in stock_details_by_idx:
                stock_details_by_idx[stock_idx] = []
            stock_details_by_idx[stock_idx].append(detail)

        if not stock_details_by_idx:
            return all_placement_details

        # Find last stock index
        last_stock_idx = all_placement_details[-1]["stock_idx"]
        last_sheet_details = stock_details_by_idx[last_stock_idx]

        # List all items on the last sheet
        products_to_place_dict = {}
        for detail in last_sheet_details:
            size = (detail["size"][0], detail["size"][1])
            if size in products_to_place_dict:
                products_to_place_dict[size] += 1
            else:
                products_to_place_dict[size] = 1

        products_to_place = [(w, h, q) for (w, h), q in products_to_place_dict.items()]

        # Save stock index was used
        used_stock_idx = {detail["stock_idx"] for detail in all_placement_details}

        best_fit_sheet = None
        best_fit_details = []
        
        # Calculate waste on the last sheet
        last_stock = next((stock for stock in stocks if stock[0] == last_stock_idx), None)
        if not last_stock:
            return all_placement_details  
        
        last_stock_width, last_stock_height = last_stock[1], last_stock[2]
        total_used_area = sum(detail["size"][0] * detail["size"][1] for detail in last_sheet_details)

        min_waste_area = last_stock_width * last_stock_height - total_used_area

        # Find the only one new best last sheet
        for idx, (stock_idx, stock_width, stock_height, _) in enumerate(stocks):
            if stock_idx in used_stock_idx:
                continue

            # Used cutting stock with only one stock
            num, sheets, details, waste_area = self.cutting_stock_2d_min_waste(
                [(stock_width, stock_height, 1)], products_to_place)

            # Check if can place all items and better min waste
            if len(details) == sum(item[2] for item in products_to_place) and waste_area < min_waste_area:
                best_fit_sheet = (stock_idx, stock_width, stock_height)
                best_fit_details = details
                min_waste_area = waste_area

        # If exist new last sheet, update
        if best_fit_sheet:
            all_placement_details = [d for d in all_placement_details if d["stock_idx"] != last_stock_idx]

            for detail in best_fit_details:
                detail["stock_idx"] = best_fit_sheet[0]

            all_placement_details.extend(best_fit_details)
        # Else, do not change

        return all_placement_details

    def _split_free_rectangle(self, stock_idx, x, y, w, h):
        new_free_rectangles = []
        for rect in self.free_rectangles[stock_idx]:
            if not self._rects_overlap(x, y, w, h, rect["x"], rect["y"], rect["w"], rect["h"]):
                new_free_rectangles.append(rect)
            else:
                self._split_rectangles(rect, x, y, w, h, new_free_rectangles)
        self.free_rectangles[stock_idx] = self._prune_free_rectangles(new_free_rectangles)

    def _split_rectangles(self, rect, x, y, w, h, new_free_rectangles):
        if rect["x"] < x:
            new_rect = {
                "x": rect["x"],
                "y": rect["y"],
                "w": x - rect["x"],
                "h": rect["h"]
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

        if x + w < rect["x"] + rect["w"]:
            new_rect = {
                "x": x + w,
                "y": rect["y"],
                "w": (rect["x"] + rect["w"]) - (x + w),
                "h": rect["h"]
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

        if rect["y"] < y:
            new_rect = {
                "x": rect["x"],
                "y": rect["y"],
                "w": rect["w"],
                "h": y - rect["y"]
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

        if y + h < rect["y"] + rect["h"]:
            new_rect = {
                "x": rect["x"],
                "y": y + h,
                "w": rect["w"],
                "h": (rect["y"] + rect["h"]) - (y + h)
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

    def _prune_free_rectangles(self, free_rectangles):
        pruned = []
        for rect in free_rectangles:
            if not any(self._is_rect_inside(rect, other) 
                       for other in free_rectangles if rect != other):
                pruned.append(rect)
        return pruned

    def _rects_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1)

    def _is_rect_inside(self, inner, outer):
        return (inner["x"] >= outer["x"] and
                inner["y"] >= outer["y"] and
                inner["x"] + inner["w"] <= outer["x"] + outer["w"] and
                inner["y"] + inner["h"] <= outer["y"] + outer["h"])

    def _is_new_episode(self, stocks):
        if self.previous_stock_states is None:
            return True
        return all(np.all(stock <= -1) for stock in stocks)

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h
    # Student code here
    # You can add more functions if needed