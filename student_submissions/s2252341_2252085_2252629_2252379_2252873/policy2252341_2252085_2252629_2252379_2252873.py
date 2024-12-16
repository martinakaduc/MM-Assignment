from policy import Policy
import numpy as np

import time
Imported_time_lib = True

Global_policy = None
class Policy2252341_2252085_2252629_2252379_2252873(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
   
        # Student code here
        global Global_policy
        if policy_id == 1:
            Global_policy = FFDH_Policy()
        elif policy_id == 2:
            Global_policy = Branch_and_Bound_Policy()
  
    def get_action(self, observation, info):
        # Student code here
        global Global_policy;
        if Global_policy==None:
            #print("running\n")  
            return 0;
        return Global_policy.get_action(observation, info);
        pass
    # Student code here   
    # You can add more functions if needed  
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
class FFDH_Policy(Policy):
    def __init__(self):
        self.init = True   
        self.level = [[]]  

    def get_action(self, observation, info):
        list_prods = observation["products"]
        sorted_prods = sorted(list_prods, key= lambda x: x['size'][1], reverse= True)

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        if self.init:
            i = 0
            while i < len(observation["stocks"]) - 1:
                self.level.append([])
                i += 1

            self.init = False

        for j,prod in enumerate(sorted_prods):
            if prod["quantity"] > 0:
                if j==len(sorted_prods)-1 and prod["quantity"]==1: #sorted_prods.index(prod)
                    self.init = True
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None

                    if not self.level[i]:
                        self.level[i].append([prod_w, prod_h])
                        pos_x, pos_y = 0, 0

                    else:
                        index_level = 0
                        previous_total_level = 0
                        curr_level = self.level[i][0][1]
                        not_inserted = True
                        while index_level < len(self.level[i]):
                            if prod_h <= curr_level:
                                if self.level[i][index_level][0] + prod_w <= stock_w:
                                    pos_x, pos_y = self.level[i][index_level][0], previous_total_level
                                    self.level[i][index_level][0] += prod_w
                                    not_inserted = False
                                    break

                            index_level += 1

                            if index_level < len(self.level[i]):
                                previous_total_level += curr_level
                                curr_level = self.level[i][index_level][1]

                        if not_inserted:
                            if previous_total_level + curr_level + prod_h <= stock_h:
                                self.level[i].append([prod_w, prod_h])
                                pos_x, pos_y = 0, previous_total_level + curr_level

                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        if self.init==True:   
            self.__init__()  
        
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
total_waste_across_episodes = 0
total_filled_ratio = 0
start_time = 0
NUM_EPISODES = 0
total_time = 0

class evaluate_class:
    def __init__(self):
        pass
    def calculate_waste(self, stocks):
            waste = 0
            for stock in stocks:
                if np.any(stock >= 0):
                    waste += np.sum(stock == -1)
            return waste
    def evaluate_init(self, nums):
        global total_waste_across_episodes, total_filled_ratio, start_time, NUM_EPISODES,total_time
        total_waste_across_episodes = 0
        total_filled_ratio = 0
        if Imported_time_lib==True:
            start_time = time.time()
        NUM_EPISODES = nums
        total_time = 0
    def evaluate_step(self, observation,info):
        global total_waste_across_episodes, total_filled_ratio, start_time,total_time
        waste = self.calculate_waste(observation["stocks"])
        print(f"Waste: {waste}")
        total_waste_across_episodes += waste
        filled_ratio = info.get("filled_ratio", 0)
        total_filled_ratio += filled_ratio
        print(info)
        if Imported_time_lib==True:
            time_temp = time.time() - start_time
            total_time += time_temp
            print(f"Execution time: {time_temp}")
            start_time = time.time()
    def evaluate_end(self):
        global total_waste_across_episodes, total_filled_ratio, start_time, NUM_EPISODES,total_time
        average_waste = total_waste_across_episodes / NUM_EPISODES
        average_filled_ratio = total_filled_ratio / NUM_EPISODES
        average_time = total_time / NUM_EPISODES
        print(f"Average Waste across {NUM_EPISODES} episodes: {average_waste:.2f}")
        print(f"Average Filled Ratio across {NUM_EPISODES} episodes: {average_filled_ratio:.2f}")
        if Imported_time_lib==True:
            print(f"Average Time across {NUM_EPISODES} episodes: {average_time:.2f}")
def Evaluate_Function(policy2210xxx,NUM_EPISODES,env,initial_inc):
    evaluate = evaluate_class()
    ep=0
    observation, info = env.reset(seed=initial_inc,options=None)
    evaluate.evaluate_init(NUM_EPISODES)###
    while ep < NUM_EPISODES:
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            evaluate.evaluate_step(observation,info)
            print("seed = ",ep+initial_inc)
            observation, info = env.reset(seed=initial_inc+ep)
            ep += 1
    evaluate.evaluate_end()
    return None
    
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
def waste_func(action_array,bin_used_array,observation):
    area_bin = 0
    area_demand = 0
    
    for n in range (len(action_array)):
        if (n%4!=0):
            continue
    
        prod_w, prod_h = observation["products"][ action_array[n] ] ["size"]
        area_demand = area_demand + prod_w*prod_h
    for n in range (len(bin_used_array)):
        if (bin_used_array[n]>=1):
            stock_w = np.sum(np.any(observation["stocks"][n] != -2, axis=1))
            stock_h = np.sum(np.any(observation["stocks"][n] != -2, axis=0))
            area_bin = area_bin + stock_w * stock_h
    return area_bin - area_demand
def per_function (demand,stock,x,y,action_array,bin_used_array,observation,quantity_array):
    global public_min, min_action_array,public_array,debug_counting
    global max_demand
    global per_function_flag
    debug_counting+=1
    #print("per_function5")
    if per_function_flag>=1:   
        return  
    if demand==-1:
        #print("enter demand ==-1")
        stock_w = np.sum(np.any(observation["stocks"][stock] != -2, axis=1))
        stock_h = np.sum(np.any(observation["stocks"][stock] != -2, axis=0))
        demand_w, demand_h = observation["products"][0]["size"]
        max_demand = len(observation["products"])
        for s in range (len(observation["stocks"])):
            for x in range (stock_w - demand_w + 1):
                for y in range (stock_h - demand_h + 1):
                    if per_function_flag>=1:
                        return
                    per_function(0,s,x,y,action_array,bin_used_array,observation,quantity_array)
        return
    if demand>=max_demand or demand<0:
        return
    demand_w, demand_h = observation["products"][demand]["size"]
    is_the_stock = False
    if bin_used_array[stock]>0:
        is_the_stock = True
        is_overlap = False
        for x_1 in range (x,x + demand_w):
            y_1 = y
            if (public_array[stock][x_1][y_1]==True or public_array[stock][x_1][y_1+demand_h-1]==True):
                return
        for y_1 in range (y,y + demand_h):
            x_1 = x
            if (public_array[stock][x_1][y_1]==True or public_array[stock][x_1+demand_w-1][y_1]==True):
                return
    else:
        waste = waste_func(action_array,bin_used_array,observation)
        if waste>=public_min:
            return 
    action_array=action_array[:]  
    bin_used_array=bin_used_array[:]  
    quantity_array=quantity_array[:]     
    action_array.append(demand)  
    action_array.append(stock)   
    action_array.append(x)    
    action_array.append(y) 
    quantity_array[demand] -= 1
    bin_used_array[stock] += demand_w*demand_h
    waste = waste_func(action_array,bin_used_array,observation)
    if waste>=public_min:
        return
    same_demand_var =  True 
    if quantity_array[demand]<=0:
        demand +=1
        same_demand_var = False
        if demand==max_demand:
            if waste<public_min:
                per_function_flag=1
                public_min = waste
                min_action_array = action_array[:]
                return   
            return  
    public_array[stock][x : x + demand_w, y : y + demand_h] = True
    input_demand_w = demand_w
    input_demand_h = demand_h
    demand_w, demand_h = observation["products"][demand]["size"]#####
    input_x = x
    input_y = y
    s=0
    while s<len(observation["stocks"]):
        if same_demand_var==True and s<stock:
            s=stock
            continue
        stock_w = np.sum(np.any(observation["stocks"][s] != -2, axis=1))   
        stock_h = np.sum(np.any(observation["stocks"][s] != -2, axis=0))  
        if bin_used_array[s]+demand_w*demand_h > stock_w*stock_h:
            s+=1
            continue
        
        for x in range (stock_w - demand_w + 1):
            if same_demand_var==True and s==stock and x<input_x:
                continue
            for y in range (stock_h - demand_h + 1):
                if public_array[s][x][y]==True:
                    continue
                if per_function_flag>=1:
                    public_array[stock][input_x : input_x + input_demand_w, input_y : input_y + input_demand_h] = False
                    return
                per_function(demand,s,x,y,action_array,bin_used_array,observation,quantity_array)
        s +=1
    public_array[stock][input_x : input_x + input_demand_w, input_y : input_y + input_demand_h] = False 
    return
    ##############################################################################################################
per_function_flag = 0
Per_pub_var = -1
class Branch_and_Bound_Policy(Policy):
    def __init__(self):
        global Per_pub_var,min_action_array,public_min,loop_max,pub_loop,max_demand,public_array,debug_counting,per_function_flag  
        Per_pub_var = -1  
        min_action_array = []     
        public_min = 99999999  
        pub_loop = 0   
        max_demand = 0    
        debug_counting=0 
        public_array = np.full((100, 100, 100), False)
        per_function_flag=0
        pass
    def get_action(self,observation,info):
        global Per_pub_var, min_action_array
        if Per_pub_var==-1:
            self.__init__()
            action = []
            bin_used = []
            quantity_array = []
            for n in range (len(observation["stocks"])):
                bin_used.append(0)
            for n in range (len(observation["products"])):
                quantity_array.append(observation["products"][n]["quantity"])
            per_function(-1,0,0,0,action,bin_used,observation,quantity_array)
            Per_pub_var = len(min_action_array) - 1 
        answer = [-1,-1,-1,-1]
        answer_idx = 3
        while answer_idx>=0:
            if Per_pub_var<0:
                Per_pub_var = -1
                break
            answer[answer_idx] = min_action_array[Per_pub_var]
            answer_idx -= 1
            Per_pub_var -= 1
        size_x, size_y = observation["products"][answer[0]]["size"]
        return {"stock_idx": answer[1], "size": [size_x,size_y], "position": (answer[2], answer[3])}
"""
import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
from student_submissions.s2252341_2252085_2252629_2252379_2252873.policy2252341_2252085_2252629_2252379_2252873 import Policy2252341_2252085_2252629_2252379_2252873,Evaluate_Function
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    #render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":

    # Uncomment the following code to test your policy
    # # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)

    policy2210xxx = Policy2252341_2252085_2252629_2252379_2252873(policy_id=2)
    for _ in range(200):
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)


    policy2210xxx = Policy2252341_2252085_2252629_2252379_2252873(policy_id=1)
    Evaluate_Function(policy2210xxx,2,env,42)
    #     if terminated or truncated:
    #         observation, info = env.reset()

env.close()
"""