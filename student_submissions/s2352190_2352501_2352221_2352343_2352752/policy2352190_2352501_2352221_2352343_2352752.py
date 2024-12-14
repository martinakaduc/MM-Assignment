from policy import Policy
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self,policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        super().__init__()
        self.patterns = []
        self.current_pattern = None
        self.current_pattern_index = 0
        self.remaining_items = {}
        self.last_successful_placement = None
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def _generate_initial_patterns(self, observation):
        items = observation["products"]
        self.remaining_items = {
            i: item["quantity"]
            for i, item in enumerate(items)
            if item["quantity"] > 0
        }

        # Sort items by area and perimeter
        sorted_items = [(i, item) for i, item in enumerate(items) if item["quantity"] > 0]
        sorted_items.sort(
            key=lambda x: (
                x[1]["size"][0] * x[1]["size"][1],  # Area
                2 * (x[1]["size"][0] + x[1]["size"][1]),  # Perimeter
                -min(x[1]["size"])  # Negative min dimension for better packing
            ),
            reverse=True
        )

        W,H=self._get_stock_size_(observation["stocks"][0])
        pattern = np.full((H, W), -1)

        # Initialize levels for level-oriented packing
        levels = [(0, H)]  # (y_position, remaining_height)

        for item_idx, item in sorted_items:
            if self.remaining_items[item_idx] <= 0:
                continue

            placed = False
            for qty in range(self.remaining_items[item_idx]):
                best_pos = None
                best_waste = float('inf')
                best_level_idx = None
                best_orientation = None

                # Try both orientations
                isize=item["size"]
                for orientation in [isize, isize[::-1]]:
                    w, h = orientation
                    item["size"]=orientation
                    if w > W or h > H:
                        continue

                    # Try placing in each level
                    for level_idx, (level_y, level_h) in enumerate(levels):
                        if h <= level_h:
                            # Try all x positions in this level
                            for x in range(W - w + 1):
                                if np.all(pattern[level_y:level_y + h, x:x + w] == -1):
                                    # Calculate waste as empty space to left and right
                                    waste = level_h - h  # Vertical waste
                                    if waste < best_waste:
                                        best_waste = waste
                                        best_pos = (x, level_y)
                                        best_level_idx = level_idx
                                        best_orientation = (w, h)

                if best_pos is not None:
                    x, y = best_pos
                    w, h = best_orientation
                    pattern[y:y + h, x:x + w] = item_idx

                    # Update levels
                    old_level_y, old_level_h = levels[best_level_idx]
                    del levels[best_level_idx]

                    # Add new level above placed item if space remains
                    if h < old_level_h:
                        levels.insert(best_level_idx, (y + h, old_level_h - h))

                    # Sort levels by y position
                    levels.sort(key=lambda x: x[0])
                    placed = True
                else:
                    break

            if not placed:
                # If we couldn't place this item, try next item
                continue

        return pattern if np.any(pattern != -1) else None

    def _update_remaining_items(self, placed_item_idx):
        if placed_item_idx in self.remaining_items:
            self.remaining_items[placed_item_idx] -= 1
            if self.remaining_items[placed_item_idx] <= 0:
                del self.remaining_items[placed_item_idx]

    def _pattern_is_depleted(self):
        """Check if current pattern has any valid placements left"""
        if self.current_pattern is None:
            return True
        for item_idx in np.unique(self.current_pattern):
            if item_idx >= 0 and item_idx in self.remaining_items:
                return False
        return True

    def get_action(self, observation, info):
        list_prods = observation["products"]

        # Initialize or regenerate patterns if needed
        if not self.patterns or self._pattern_is_depleted():
            new_pattern = self._generate_initial_patterns(observation)
            if new_pattern is not None:
                self.patterns = [new_pattern]
                self.current_pattern = new_pattern.copy()
                self.current_pattern_index = 0
                self.last_successful_placement = None

            else:
                # If we can't generate a new pattern, try with remaining items
                remaining_items = [i for i, item in enumerate(list_prods)
                                   if item["quantity"] > 0]
                if remaining_items:
                    return {
                        "stock_idx": 0,
                        "size": list_prods[remaining_items[0]]["size"],
                        "position": (0, 0)
                    }

        # Find next item to place
        if self.current_pattern is not None:
            h, w = self.current_pattern.shape
            # Start search from last successful placement if available
            start_y = 0 if self.last_successful_placement is None else self.last_successful_placement[0]

            for y in range(start_y, h):
                for x in range(w):
                    item_idx = self.current_pattern[y, x]
                    if item_idx >= 0 and item_idx in self.remaining_items:
                        item_size = list_prods[item_idx]["size"]

                        # Try to place in each stock
                        stock_idx1=None
                        stock_idx2=None
                        for stock_idx, stock in enumerate(observation["stocks"]):
                            stock_w, stock_h = self._get_stock_size_(stock)
                            for pos_y in range(stock_h - item_size[1] + 1):
                                for pos_x in range(stock_w - item_size[0] + 1):
                                    #print("ITEM_SIZE", item_size)
                                    if self._can_place_(stock, (pos_x, pos_y), item_size):
                                        self._update_remaining_items(item_idx)
                                        self.last_successful_placement = (y, x)
                                        stock_idx1=stock_idx
                                        item_size1=item_size
                                        posx1,posy1=pos_x,pos_y
                                        break
                                if(stock_idx1!=None):
                                    break
                            if(stock_idx1!=None):
                                break        
                        for stock_idx, stock in enumerate(observation["stocks"]):
                            stock_w, stock_h = self._get_stock_size_(stock)
                            item_size=item_size[::-1]
                            for pos_y in range(stock_h - item_size[1] + 1):
                                for pos_x in range(stock_w - item_size[0] + 1):
                                    #print("ITEM_SIZE", item_size)
                                    if self._can_place_(stock, (pos_x, pos_y), item_size):
                                        self._update_remaining_items(item_idx)
                                        self.last_successful_placement = (y, x)
                                        stock_idx2=stock_idx
                                        item_size2=item_size
                                        posx2,posy2=pos_x,pos_y
                                        break
                                if(stock_idx2!=None):
                                    break
                            if(stock_idx2!=None):
                                break
                        if(stock_idx2==None and stock_idx1!= None):
                                return {
                                    "stock_idx": stock_idx1,
                                    "size": item_size1,
                                    "position": (posx1,posy1)
                                }
                        if(stock_idx1==None and stock_idx2!= None):
                                return {
                                    "stock_idx": stock_idx2,
                                    "size": item_size2,
                                    "position": (posx2,posy2)
                                }                       
                        if(stock_idx1<=stock_idx2):
                            if(stock_idx1!=None):
                                return {
                                    "stock_idx": stock_idx1,
                                    "size": item_size1,
                                    "position": (posx1,posy1 )
                                }
                        if(stock_idx1>stock_idx2):
                            if(stock_idx2!=None):
                                return {
                                    "stock_idx": stock_idx2,
                                    "size": item_size2,
                                    "position": (posx2,posy2)
                                }

        # Return empty action if no placement found
        return {
            "stock_idx": 0,
            "size": [0, 0],
            "position": (0, 0)
        }
    def custom_testcase_evaluation(self, observation):

                # Tính tổng diện tích của các sản phẩm
                fitlist=[]
                count=0
                total_prod_area=0
                for i,stock in enumerate(observation["stocks"]):
                    if self.placed(stock):
                        #print("Stock idx nhap vao", i)
                        prod_area=self.get_product_area(stock)
                        total_prod_area += prod_area
                        wei,hei=self._get_stock_size_(stock)
                        stock_size=wei*hei
                        fit=1-((stock_size-prod_area)/stock_size)
                        fitlist.append(fit)
                        count +=1
                    continue
                print("Ideal result")
                widi, heii=self._get_stock_size_(observation["stocks"][0])
                istocks=widi*heii
                # print(istocks)
                # print(total_prod_area)
                sizecodinh=istocks
                icount=1
                while(istocks < total_prod_area):
                    icount+=1
                    istocks+=sizecodinh
                ifit=1-((istocks-total_prod_area)/istocks)
                # print(total_prod_area, istocks)
                print("AVERAGE IDEAL FIT: ", ifit)
                print("IDEAL STOCK NUMBER: ", icount)
                
                print("Starting evaluation")
                #print(fitlist)
                print("Best Fit: ", max(fitlist))
                print("Number of stocks", count)
                avf=(sum(fitlist)/count)
                print("Average Fit: ", avf)
                offrate=(avf/ifit)*100
                print("SO VOI KET QUA LY TUONG DAT: ",offrate,end="")
                print("%")
                return "End evaluation"


        