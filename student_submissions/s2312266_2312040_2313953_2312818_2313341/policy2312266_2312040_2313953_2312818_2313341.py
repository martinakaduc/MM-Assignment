from policy import Policy
from policy import GreedyPolicy
import random
import numpy as np
import math 
import copy

#Giai thuat nhanh-can


class Policy2312266_2312040_2313953_2312818_2313341(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        super().__init__()
        self.policy_id = policy_id

        if policy_id == 1:
            self.strategy = self.BranchAndBound()
        elif policy_id == 2:
            self.strategy = self.SimulatedAnnealingPolicy()

    def get_action(self, observation, info):
        return self.strategy.get_action(observation, info)

    class BranchAndBound(Policy):
        def __init__(self):
            self.save_area = 0
            self.save_cut = []
            self.idx = 0

        def get_action(self, observation, info):
            # Lấy danh sách stocks và demands
            stocks = observation["stocks"]
            demands = observation["products"]

            # Sắp xếp demands theo diện tích giảm dần
            demands = sorted(demands, key=lambda x: (x["size"][0] * x["size"][1]), reverse=True)   
            # Chạy giải thuật
            quantity = [demand["quantity"] for demand in demands]
            ret =  self.branch_and_bound(stocks , self.idx, demands , -1, quantity)
            if all(x <= 0 for x in quantity ):
                self.idx = 0 
                self.save_area = 0
                self.save_cut = []
            return ret


        def branch_and_bound(self, stocks, stock_idx,  remaining_demands,  idx , quantity): 
                    if all(x <= 0 for x in quantity) : 
                        self.idx = 0 
                        return {"stock_idx": -1,"size": (0,0), "position": (-1,-1)}

                    stock = stocks[stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    # TODO nếu có mảnh được thêm vào thì dừng giải thuật, để có thể chạy qua step render frame 
                    for demand_idx, demand in enumerate(remaining_demands):
                        if quantity[demand_idx] <= 0:
                            continue
                        prod_w, prod_h = demand["size"]
                        if(demand_idx < idx): continue  
                
                        for rotation in [(prod_w, prod_h), (prod_h, prod_w)]:
                            rotated_w, rotated_h = rotation
                            if rotated_w <= stock_w and rotated_h <= stock_h:
                                for x in range(stock_w - rotated_w + 1):
                                    for y in range(stock_h - rotated_h + 1):
                                        if quantity[demand_idx] <= 0: break
                                        if self._can_place_(stock, (x, y), (rotated_w, rotated_h)):
                                            demand["size"] = (rotated_w, rotated_h)
                                            self.save_cut.append({"stock_idx": stock_idx, "size": demand["size"], "position": (x, y), "prod_id": demand_idx})
                                            self.save_area = self.save_area + (prod_w * prod_h)
                                            quantity[demand_idx] -= 1
                                            # nếu idx != -1 thì quay lui 
                                            if(idx != -1):
                                                self._place_piece(stock, (x, y), (rotated_w, rotated_h), demand_idx)
                                                remaining_demands[demand_idx]["quantity"] -= 1
                                                return self.branch_and_bound(stocks,stock_idx, remaining_demands, idx, quantity)
                                            return {"stock_idx": stock_idx, "size": demand["size"], "position": (x, y)}     
                                if(quantity[demand_idx] <= 0): break

                    # Nếu không có cái nào lắp vào được, ta thử xóa mảnh phía trước để lắp các mảnh nhỏ hơn vào
                    # Điều kiện dừng
                    if np.all(stock[0:stock_w  , 0:stock_h ] != -1): #Nếu lắp đầy stock[i] thì dừng
                        self.idx += 1
                        return {"stock_idx": -1, "size": (0, 0), "position": (-1, -1)}
                    if((not self.save_cut) or self.save_cut[-1]["stock_idx"] != stock_idx): # Nếu pop hết các miếng lắp cho stocks[i] thì dừng
                        self.idx += 1
                        return {"stock_idx": -1, "size": (0, 0), "position": (-1, -1)}
                    if(idx == len(remaining_demands)): # Nếu quay lui đến miếng prod nhỏ nhất mà vẫn ko lắp vừa thì dừng
                        self.idx += 1
                        return {"stock_idx": -1, "size": (0, 0), "position": (-1, -1)}
                    #lưu thông tin lại để cập nhật sau 
                    temp_stocks = [copy.deepcopy(stock) for stock in stocks]
                    temp_area = self.save_area
                    temp_cut = copy.deepcopy(self.save_cut)
                    temp_quantity = copy.deepcopy(quantity)
                    temp_remaining_demands = copy.deepcopy(remaining_demands)

                    next_idx = self.save_cut[-1]["prod_id"] + 1
                    old_pos = self.save_cut[-1]["position"]
                    old_size = self.save_cut[-1]["size"]
                    #xóa mảnh phía trước
                    self._remove_piece(temp_stocks[stock_idx], old_pos, old_size)
                    self.save_cut.pop() 

                    quantity[next_idx -1] += 1
                    remaining_demands[next_idx - 1]["quantity"] += 1
                    self.save_area -= old_size[0] * old_size[1]
                    self.branch_and_bound(temp_stocks,stock_idx, remaining_demands, next_idx, quantity)
                    # check xem nếu như save_area trong quá trình quay lui xấu hơn cái vừa rồi thì đặt lại
                    if(self.save_area < temp_area):   
                        self.save_area = temp_area
                        self.save_cut = temp_cut
                        quantity = temp_quantity
                        remaining_demands = temp_remaining_demands
                    else:
                        stocks[stock_idx][:] = temp_stocks[stock_idx]
                    return {"stock_idx": -1, "size": (0, 0), "position": (-1, -1)}

        # method đặt các product vào stock
        def _place_piece(self, stock, position, size, demand_idx):
                x, y = position
                w, h = size
                stock[x : x + w, y : y + h] = demand_idx
        # method xóa các product khỏi stock
        def _remove_piece(self, stock, position, size):
                x, y = position
                w, h = size
                stock[x : x + w, y : y + h] = -1

        #Giải thuật luyện thép
    class SimulatedAnnealingPolicy(Policy):
            def __init__(self, initial_temperature=1000, cooling_rate=0.95, iterations=1000):
                # Khởi tạo các tham số của giải thuật Simulated Annealing
                self.initial_temperature = initial_temperature  # Nhiệt độ ban đầu
                self.cooling_rate = cooling_rate                # Hệ số làm nguội
                self.iterations = iterations                    # Số lần lặp

            def get_action(self, observation, info):
                # Khởi tạo các biến cho giải thuật
                best_action = None
                best_cost = float('inf')
                temperature = self.initial_temperature

                # Sử dụng giải thuật tham ăn để tạo giải pháp ban đầu
                greedy_action = GreedyPolicy().get_action(observation, info) 
                if greedy_action is None:
                    return None                 # Trường hợp không tìm thấy giải pháp
        
                # Gán giải pháp ban đầu cho giải thuật
                current_action = greedy_action
                current_cost = self.calculate_cost(current_action, observation) 

                # Giải thuật Simulated Annealing để tìm ra giải pháp tốt nhất 
                # dựa trên các giải pháp láng giềng
                for _ in range(self.iterations):
                    # Tạo giải pháp láng giềng
                    neighbor_action = self.generate_neighbor(current_action, observation)
                    if neighbor_action is None:
                      continue          # Bỏ qua lần lặp này nếu không có

                    # Tính toán sự khác biệt về chi phí
                    neighbor_cost = self.calculate_cost(neighbor_action, observation)
                    delta_cost = neighbor_cost - current_cost

                    # Điều kiện chấp nhận láng giềng
                    if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                        current_action = neighbor_action
                        current_cost = neighbor_cost

                    # Cập nhật giải pháp tốt nhất
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_action = current_action

                    # Giảm nhiệt độ
                    temperature *= self.cooling_rate

                return best_action


            def generate_neighbor(self, action, observation):
                # Tạo giải pháp láng giềng bằng cách thay đổi vị trí của một sản phẩm ngẫu nhiên
                new_action = action.copy()
                new_x = max(0, action["position"][0] + random.randint(-1,1))
                new_y = max(0, action["position"][1] + random.randint(-1,1))

                new_action["position"] = (new_x, new_y)

                # Kiểm tra vị trí có hợp lý?
                stock = observation["stocks"][new_action["stock_idx"]]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = new_action["size"]
                if new_x + prod_w <= stock_w and new_y + prod_h <= stock_h and self._can_place_(stock, new_action["position"], new_action["size"]):
                    return new_action
                else:
                    return None # Trả về None nếu không có giải pháp láng giềng hợp lý


            def calculate_cost(self, action, observation):
                # Tính chi phí dựa trên diện tích còn dư
                stock = observation["stocks"][action["stock_idx"]] # Lấy stock
                prod_area = action["size"][0] * action["size"][1]  # Diện tích sản phẩm
                stock_area = np.sum(stock != -1)  # Tính diện tích đã sử dụng trên stock
                return stock_area - prod_area  # Chi phí là diện tích còn dư
        