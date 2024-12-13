from policy import Policy
from student_submissions.s2210xxx.binmanager import BinManager
from student_submissions.s2210xxx.item import Item



class MaximalRectangle_bestfit(Policy):


    def __init__(self):
            self.bin_current_idx = 0
            self.item_idx_to_bin_idx = 0
            pass
    

    def get_stocks_list(self, observation):
        stocks_list = [
            (i,self._get_stock_size_(stock))
            for i, stock in enumerate(observation["stocks"])
        ]
        stocks_list.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)  # Calculate area as width * height
        return stocks_list
    
    
    def get_items_list(self, observation):
        items = []
        for product in observation["products"]:
            for _ in range(product["quantity"]):
                width, height = product["size"]
                items.append(Item(width, height))
        return items
    




    def get_action(self, observation, info):#After execute binmanager, get the action each bin.
        # Student code here
        if(info["filled_ratio"]==0):
            self.bin_current_idx = 0
            self.item_idx_to_bin_idx = 0
            stocks_list = self.get_stocks_list(observation)
            items_list = self.get_items_list(observation)
            self.binmanager = BinManager(10, 6,bin_algo='bin_best_fit',pack_algo='guillotine', heuristic='best_shortside', rotation=True, sorting=True,unused_bins=stocks_list)
            self.binmanager.add_items(*items_list)
            self.binmanager.execute()
            


        if self.bin_current_idx < len(self.binmanager.bins):
            bin = self.binmanager.bins[self.bin_current_idx]
            if self.item_idx_to_bin_idx < len(bin.items):
                item = bin.items[self.item_idx_to_bin_idx]
                self.item_idx_to_bin_idx += 1
                # return item.width, item.height, self.bin_current_idx
                # print(item.width, item.height,item.x,item.y ,self.bin_current_idx)
                return {"stock_idx": bin.idx, "size": [item.width,item.height], "position": (item.x, item.y)}
            else:
                self.bin_current_idx += 1
                self.item_idx_to_bin_idx = 0
                return self.get_action(observation, info)
        else:
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}            


    # Student code here
    # You can add more functions if needed
