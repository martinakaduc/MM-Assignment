# from policy import Policy





# def Policy2210xxx(Policy):
#     def __init__(self):
#         # Student code here
#         pass

#     def get_action(self, observation, info):
#         # Student code here
#         pass

#     # Student code here
#     # You can add more functions if needed



from policy import Policy
from .MaximalRectanglePolicy import MaximalRectangle_bestfit
from .ColumnGenerationPolicy import ColumnGenerationPolicy



class Policy2312183_2311687_2311681_2311813_2313751(Policy):


    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id

        if policy_id == 1:
            self.current_policy = MaximalRectangle_bestfit()
        elif policy_id == 2:
            self.current_policy = ColumnGenerationPolicy()


    def get_action(self, observation, info):
        return self.current_policy.get_action(observation, info)
    

    













# from policy import Policy





# def Policy2210xxx(Policy):
#     def __init__(self):
#         # Student code here
#         pass

#     def get_action(self, observation, info):
#         # Student code here
#         pass

#     # Student code here
#     # You can add more functions if needed



# from policy import Policy
# from student_submissions.s2210xxx.binmanager import BinManager
# from student_submissions.s2210xxx.item import Item



# class Policy2210xxx(Policy):


#     def __init__(self, policy_id=1):
#         assert policy_id in [1, 2], "Policy ID must be 1 or 2"

#         # Student code here
#         self.policy_id = policy_id

#         if policy_id == 1:
#             self.bin_current_idx = 0
#             self.item_idx_to_bin_idx = 0
#             pass
#         elif policy_id == 2:
#             pass
    

#     def get_stocks_list(self, observation):
#         stocks_list = [
#             (i,self._get_stock_size_(stock))
#             for i, stock in enumerate(observation["stocks"])
#         ]
#         stocks_list.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)  # Calculate area as width * height
#         return stocks_list
    
    
#     def get_items_list(self, observation):
#         items = []
#         for product in observation["products"]:
#             for _ in range(product["quantity"]):
#                 width, height = product["size"]
#                 items.append(Item(width, height))
#         return items
    




#     def get_action(self, observation, info):
#         if self.policy_id == 1:
#             return self.get_action_1(observation, info)
#         elif self.policy_id == 2:
#             return None

#     def get_action_1(self, observation, info):#After execute binmanager, get the action each bin.
#         # Student code here
#         if(info["filled_ratio"]==0):
#             self.bin_current_idx = 0
#             self.item_idx_to_bin_idx = 0
#             stocks_list = self.get_stocks_list(observation)
#             items_list = self.get_items_list(observation)
#             self.binmanager = BinManager(10, 6,bin_algo='bin_best_fit',pack_algo='guillotine', heuristic='best_shortside', rotation=True, sorting=True,unused_bins=stocks_list)
#             self.binmanager.add_items(*items_list)
#             self.binmanager.execute()
            


#         if self.bin_current_idx < len(self.binmanager.bins):
#             bin = self.binmanager.bins[self.bin_current_idx]
#             if self.item_idx_to_bin_idx < len(bin.items):
#                 item = bin.items[self.item_idx_to_bin_idx]
#                 self.item_idx_to_bin_idx += 1
#                 # return item.width, item.height, self.bin_current_idx
#                 # print(item.width, item.height,item.x,item.y ,self.bin_current_idx)
#                 return {"stock_idx": bin.idx, "size": [item.width,item.height], "position": (item.x, item.y)}
#             else:
#                 self.bin_current_idx += 1
#                 self.item_idx_to_bin_idx = 0
#                 return self.get_action(observation, info)
#         else:
#             prod_size = [0, 0]
#             stock_idx = -1
#             pos_x, pos_y = 0, 0
#             return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}            


#     # Student code here
#     # You can add more functions if needed

