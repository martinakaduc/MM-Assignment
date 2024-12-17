from policy import Policy
import random
from gym_cutting_stock.envs.cutting_stock import CuttingStockEnv
import copy
import numpy as np


class Policy2312332_2312197_2311957_2312949(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = FirstFitDecreasingPolicy()
        elif policy_id == 2:
            self.policy = AnnealingPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed

class FirstFitDecreasingPolicy(Policy):
    def __init__(self):
        self.done_stock=[]
        self.sorted_stock=[]
        self.sorted_prods=[]
        pass

    def get_action(self, observation, info, check = False, rev = True):
        if info["filled_ratio"] == 0:
            self.done_stock = []
            self.sorted_stock=[]
            self.sorted_prods=[]
        if check:
            self.done_stock = []
            self.sorted_stock=[]
            self.sorted_prods=[]   
        if len(self.sorted_stock) == 0:
            list_stocks = observation["stocks"]
            self.stock_sorted = sorted(
                [(i, stock) for i, stock in enumerate(list_stocks)],
                key=lambda x: ((self._get_stock_size_(x[1])[0]* self._get_stock_size_(x[1])[1]), max(self._get_stock_size_(x[1]))),
                reverse = rev
            )
            
        if len(self.sorted_prods) == 0:
            list_prods = observation["products"]
            self.sorted_prods = sorted(
                [(i, prod) for i, prod in enumerate(list_prods) if prod["quantity"] != 0],
                key=lambda x: (x[1]["size"][0]* x[1]["size"][1], x[1]["size"][0]),
                reverse=True
            )
        list_prods = observation["products"]
        
        remaining_prods = [prod[1] for prod in self.sorted_prods if list_prods[prod[0]]["quantity"] != 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        s_stock = [stock for stock in self.stock_sorted if stock[0] not in self.done_stock]
        while remaining_prods:
            for stock in s_stock: 
                for prod in remaining_prods:                 
                    stock_w, stock_h = self._get_stock_size_(stock[1])
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    if stock_w >= prod_w and stock_h >= prod_h:   
                        pos_x, pos_y= self._find_position_(stock[1], [prod_w, prod_h])
                        if pos_x is not None and pos_y is not None:
                            stock_idx = stock[0]
                            return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": [pos_x, pos_y]}
            
                    stock_w, stock_h = self._get_stock_size_(stock[1])
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size[1], prod_size[0]
                    if stock_w >= prod_w and stock_h >= prod_h:
                                    pos_x, pos_y = self._find_position_(stock[1], (prod_w, prod_h))
                                    if pos_x is not None and pos_y is not None:
                                            stock_idx = stock[0]
                                            return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": [pos_x, pos_y]}


                self.done_stock.append(stock[0])
        
        prod_size=[0, 0]
        return {"stock_idx": stock_idx, "size": prod_size, "position": [pos_x, pos_y]}

    def _find_position_(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        pos = [(i, j) for i, row in enumerate(stock) for j, s in enumerate(row) if s == -1 and i  <= stock_w - prod_w and j <= stock_h - prod_h]
        #print(pos)

        for x in pos:
            if self._can_place_(stock, x, prod_size):
                return x[0], x[1]
       

        return None, None
    
class AnnealingPolicy(Policy):
    def __init__(self):
        self.cur=[]
        self.best=0
        self.cnt=0
        self.doneS=[]
        self.rev = False
        pass
    def setStart(self, observation, info, rev):
        env = subCut()
        ar=[]        
        subO = observation
        subI = info
        ob, inf = env.reset(subO, subI)
        ff_policy = FirstFitDecreasingPolicy();
        while True:
            action = ff_policy.get_action(ob, inf, False, rev)
            ob, reward, terminated, truncated, inf = env.step(action)
            if action["stock_idx"] != -1:
                ar.append(action)
            else:
                print("dunno")
            if terminated or truncated:
                break; 
        env.close()
        return inf, ar
        
    def get_action(self, observation, info):
        if info["filled_ratio"] == 0:
            env = subCut()
            zob = copy.deepcopy(observation)
            zinfo = copy.deepcopy(info)
            self.cur = []
            self.best = 0
            self.cnt = 0
            sInfo, arr = self.setStart(zob, zinfo, False)
            self.cur = arr
            filled_ratio = sInfo.get("filled_ratio", 1e-6) 
            trim_loss = sInfo.get("trim_loss", 1e-6) 
            
            self.best =1.0/trim_loss*1000 + 1.0/filled_ratio*10
            sInfo, arr = self.setStart(zob, zinfo, True)
            filled_ratio = sInfo.get("filled_ratio") 
            trim_loss = sInfo.get("trim_loss") 
            sub = 1.0/trim_loss*1000 + 1.0/filled_ratio * 10
            if sub > self.best:
                self.best = sub
                self.rev = True
                self.cur = arr
            no_improve = 0
            env = subCut()
            Temp = 100
            hat = 0
            while(no_improve <=20 and Temp > 0.0001):
                ob = copy.deepcopy(zob)
                Temp = Temp * (0.995 ** hat)
                ob, subinfo = env.reset(ob, zinfo)
                arrayFill = []        
                one = 0
                two = 0   
                choice = random.randint(0, len(self.cur))
                if choice % 2 == 0:
                    cnt = 0
                    while self.cur[one]["size"][0] == self.cur[two]["size"][0] and self.cur[one]["size"][1] == self.cur[two]["size"][1]:
                        cnt+=1
                        if cnt > 40:
                            break
                        if len(self.cur) <2:
                            break
                        one, two = random.sample(range(0, len(self.cur)), 2)
                        three = random.randint(0, 100)    
                        if (self.cur[one]["size"][0] != self.cur[two]["size"][0]) and (self.cur[one]["size"][1] != self.cur[two]["size"][1]) and self.cur[one]["stock_idx"]!=-1 and self.cur[two]["stock_idx"]!=-1:
                            self.cur[one]["size"], self.cur[two]["size"] = self.cur[two]["size"], self.cur[one]["size"]
                            break
                else:
                    one, two = random.sample(range(0, len(self.cur)), 2)
                    three = random.randint(0, len(self.cur) // 2)
                    if three % 2 == 0:
                        self.cur[one]["size"][0], self.cur[one]["size"][1] = self.cur[one]["size"][1], self.cur[one]["size"][0]  
                    three = random.randint(0, len(self.cur) // 2)
                    if three % 2 == 0:
                        self.cur[two]["size"][0], self.cur[two]["size"][1] = self.cur[two]["size"][1], self.cur[two]["size"][0]  
                    
                for cur in self.cur:                                 
                    if self._can_place_(ob["stocks"][cur["stock_idx"]], cur["position"], cur["size"]):   
                        ob, reward, terminated, truncated, subinfo = env.step(cur)
                        arrayFill.append(cur)
                ff_policy = FirstFitDecreasingPolicy();       
                while True:
                    three = random.randint(0, len(self.cur))
                    action = ff_policy.get_action(ob, subinfo, True, three %2 == 1)
                    ob, reward, terminated, truncated, subinfo = env.step(action)
                    if action["stock_idx"] != -1:
                        arrayFill.append(action) 
                    if terminated or truncated:
                        break    

                filled_ratio = subinfo.get("filled_ratio") 
                trim_loss = subinfo.get("trim_loss") 
                val =  1.0/trim_loss*1000 + 1.0/filled_ratio * 10
                E = val - self.best
                if E > 0:
                    self.best = val
                    self.cur = arrayFill
                    no_improve = 0
                else:
                    if random.uniform(0,1) < np.exp(E/Temp):
                        self.best = val
                        self.cur = arrayFill
                        break
                    no_improve += 1 
                hat+=1
            env.close()
        while self.cur:
            return self.cur.pop(0)
        env = subCut()
        thisOb, thisInfo = env.reset(observation , info)
        ff_policy = FirstFitDecreasingPolicy();
        return ff_policy.get_action(thisOb, thisInfo, True, self.rev)
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

class subCut(CuttingStockEnv):    

    def reset(self, observation, info, render = None):
        super().reset(42)
        self.render_mode = render
        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        self._stocks = copy.deepcopy(observation["stocks"])  
        self._products = copy.deepcopy(observation["products"])  
        self._stocks = tuple(self._stocks)
        self._products = tuple(self._products)
        # Cập nhật thông tin
        ob = self._get_obs()
        info = copy.deepcopy(self._get_info())

        return ob, info

