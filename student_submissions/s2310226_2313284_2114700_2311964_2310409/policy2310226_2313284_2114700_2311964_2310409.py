from policy import Policy




class Policy2310226_2313284_2114700_2311964_2310409(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy = None
        
        # Khởi tạo policy dựa trên policy_id
        if policy_id == 1:
            self.policy = Greedy()
        elif policy_id == 2:
            self.policy = FFD(policy_id=2)

    def get_action(self, observation, info):
        # Gọi phương thức get_action của chính sách tương ứng
        if self.policy:
            return self.policy.get_action(observation, info)
        else:
            raise ValueError("No policy has been initialized.")



class Greedy(Policy):
    def __init__(self):
        self.first_action=False 
    
    def get_action(self, observation,info):
        
        list_stock=observation["stocks"]
        list_prod=observation["products"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        big_stock_idx= self.bis_stock(observation)
        big_prod_size= self.big_prod(observation)
        
        for i in big_prod_size:
            prod=list_prod[i]
            if prod["quantity"] <= 0:
                continue
            prod_size = prod["size"]
            for j in big_stock_idx:
                stock=list_stock[j]
                prod_w, prod_h = prod_size
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w* stock_h < prod_h* prod_w:
                    continue
                if stock_w >= prod_h and stock_h >= prod_w:  # Nếu sản phẩm xoay ngang có thể vừa
                    prod_size = self.rotate_prod(prod_size)
                    prod_w, prod_h = prod_size
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            pos_x, pos_y = x, y
                            return {"stock_idx": j, "size": prod_size, "position": (pos_x, pos_y)}

    def rotate_prod(self, prod_size):
        prod_w,prod_h=prod_size
        return prod_h, prod_w

    def bis_stock(self, observation):
        list_stock=observation["stocks"]
        
        return self.arrange_stock(list_stock)
    
    def big_prod(self, observation):
        list_prod= observation["products"]
        return self.arrange_prod(list_prod)
    
    def arrange_prod(self, list_prod):
        prod_arrange = []
        for i in range(len(list_prod)):
            prod_size= list_prod[i]["size"]
            prod_w, prod_h = prod_size
            area = prod_w * prod_h
            prod_arrange.append((area, i))  # Lưu diện tích và chỉ số vào tuple
        
        # Sắp xếp danh sách theo diện tích (giảm dần)
        prod_arrange.sort(reverse=True, key=lambda x: x[0])
        
        # Tạo danh sách các stock mới theo chỉ số đã sắp xếp
        new_list_prod = [i for _, i in prod_arrange]
        
        return new_list_prod
    
    def arrange_stock(self, list_stock):        
        # Tạo một danh sách các tuple (diện tích, chỉ số)
        stock_arrange = []
        for i in range(len(list_stock)):
            stock_w, stock_h = self._get_stock_size_(list_stock[i])
            area = stock_w * stock_h
            stock_arrange.append((area, i))  # Lưu diện tích và chỉ số vào tuple
        
        # Sắp xếp danh sách theo diện tích (giảm dần)
        stock_arrange.sort(reverse=True, key=lambda x: x[0])
        
        # Tạo danh sách các stock mới theo chỉ số đã sắp xếp
        new_list_stock = [i for _, i in stock_arrange]
        
        return new_list_stock
    
    
class  FFD(Policy):
    def __init__(self,policy_id=1):
        self.first_action=False 
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id 
    
    def get_action(self, observation,info):
        
        # Get the list of products and sort them by area in decreasing order
        list_prods = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Iterate over sorted products
        for prod in list_prods:
            prod_size = prod["size"]
            pos_x, pos_y = None, None

            # Iterate through stocks
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Check both orientations of the product
                for orientation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
                    prod_w, prod_h = orientation
                    if stock_w >= prod_w and stock_h >= prod_h:
                        # Try to place the product in the first available position
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                    pos_x, pos_y = x, y
                                    prod_size = [prod_w, prod_h]
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                    if pos_x is not None and pos_y is not None:
                        break

                if pos_x is not None and pos_y is not None:
                    break

            # If a valid position is found, stop searching
            if pos_x is not None and pos_y is not None:
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
