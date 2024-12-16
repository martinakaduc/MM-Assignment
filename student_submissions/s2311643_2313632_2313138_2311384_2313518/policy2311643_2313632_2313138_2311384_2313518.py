from policy import Policy


class Policy2311643_2313632_2313138_2311384_2313518(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.max_height = [0] * 100
            self.curx = [0] * 100
            self.cury = [0] * 100
            self.level = [[[0, 0] for _ in range(30)] for _ in range(100)]
            self.count_lvl= [0]*100
            self.currentlvl = [0]*100
        elif policy_id == 2:
            self.max_height = [0] * 100
            self.curx = [0] * 100
            self.cury = [0] * 100
    def sort_stock(self,observation):
        area_list = []
        for i,stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            area_list.append((stock_w*stock_h,i))
        area_list_sorted = sorted(area_list, key=lambda x: x[0], reverse=True)
        index_sorted = [x[1] for x in area_list_sorted]
        return index_sorted
    def sort_product(self,observation):
        area_list = []
        list_prod = observation["products"]
        sum = 0
        for i,prod in enumerate(list_prod):
            prod_size = prod["size"]
            sum+= prod["quantity"]
            prod_w, prod_h = prod_size
            area_list.append((prod_w,prod_h,i))
        area_list_sorted = sorted(area_list, key=lambda x: x[1], reverse=True)
        index_sorted = [x[2] for x in area_list_sorted]
        return index_sorted,sum
    def get_action(self, observation, info):
        # Student code here
        if(self.policy_id==1):
            index_sorted_stock = self.sort_stock(observation)
            index_sorted_prod, numprod = self.sort_product(observation)
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            for i in index_sorted_prod:
                product = observation["products"][i]
                if product["quantity"]>0:
                    prod_size = product["size"]
                    prod_w,prod_h = prod_size
                    if prod_h == 0 and prod_w == 0:
                        continue
                    for j in index_sorted_stock:
                        stock = observation["stocks"][j]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        if(prod_h>stock_h or prod_w > stock_w):
                            continue
                        for k in range (len(self.level[j])):
                            # kiểm tra product có thể đặt vào vị trí nào ở level thứ count nào không
                            levelj = self.level[j]
                            if(levelj[k][0]==0 and levelj[k][1]==0):
                                break
                            if self._can_place_(stock,(levelj[k][0],levelj[k][1]),prod_size) and levelj[k][0]+prod_w < stock_w and levelj[k][1]+prod_h < stock_h:
                                self.level[j][self.currentlvl[j]][0] = self.curx[j]         #Cập nhật tọa độ x của con trỏ tại level cũ
                                self.level[j][self.currentlvl[j]][1] = self.cury[j]         #Cập nhật tọa độ y của con trỏ tại level cũ
                                self.curx[j] = levelj[k][0]                                 #Nhảy lên tọa độ x của con trỏ tại level mới
                                self.cury[j] = levelj[k][1]                                 #Nhảy lên tọa độ x của con trỏ lại level mới
                                self.currentlvl[j] = k                                      #Cập nhật lại level đang xếp product
                                break
                        # Case không đủ chiều rộng thì mở level mới
                        if self.curx[j]+prod_w > stock_w:
                            # Nếu mở level mới cũng không đủ không gian thì qua stock mới
                            if self.max_height[j]+prod_h>stock_h:
                                continue
                            # Mở level mới
                            self.level[j][self.count_lvl[j]][0] = self.curx[j]      #Cập nhật tọa độ x của con trỏ tại level đang xếp product
                            self.level[j][self.count_lvl[j]][1] = self.cury[j]      #Cập nhật tọa độ y của con trỏ tại level đang xếp product
                            self.count_lvl[j]+=1                                    #Tăng số lớp của stock thứ j lên 1
                            self.curx[j] = 0                                        #Đưa con trỏ x trở về đầu stock_w
                            self.cury[j] = self.max_height[j]                       #Đưa con trỏ y lên vị trị cao nhất để mở ra level mới
                            self.currentlvl[j]=self.count_lvl[j]                    #Cập nhật lại level đang xếp product
                        pos_x = self.curx[j]
                        pos_y = self.cury[j]
                        stock_idx = j
                        self.curx[j] = (self.curx[j] + prod_w)
                        self.max_height[j]=max(self.max_height[j],self.cury[j]+prod_h)
                        break
                    break
            # reset các mảng lưu trữ về giá trị ban đầu nếu hết products
            if(numprod==1):
                self.max_height = [0] * 100
                self.curx = [0] * 100
                self.cury = [0] * 100
                self.level = [[[0, 0] for _ in range(30)] for _ in range(100)]
                self.count_lvl= [0]*100
                self.currentlvl = [0]*100
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        elif(self.policy_id==2):
            # lấy danh sách thứ tự các stock theo diện tích giảm dần
            index_sorted_stock = self.sort_stock(observation)
            # lấy danh sách thứ tự các product theo chiều cao giảm dần
            index_sorted_prod, numprod = self.sort_product(observation)
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            for i in index_sorted_prod:
                product = observation["products"][i]
                if product["quantity"]>0:
                    prod_size = product["size"]
                    prod_w,prod_h = prod_size
                    if prod_h == 0 and prod_w == 0:
                        continue
                    for j in index_sorted_stock:
                        stock = observation["stocks"][j]
                        stock_w, stock_h = self._get_stock_size_(stock) 
                        if(prod_h>stock_h or prod_w > stock_w):
                            continue
                        # case chiều rộng không đủ thì mở ra level mới
                        if self.curx[j]+prod_w > stock_w:
                            # Nếu mở ra level mới nhưng chiều cao vẫn không đủ thì chuyển qua stock mới
                            if self.max_height[j]+prod_h>stock_h:
                                continue
                            self.curx[j] = 0
                            self.cury[j] = self.max_height[j]
                        # case chiều rộng vẫn đủ nhưng chiều cao không đủ thì chuyển qua stock mới
                        if self.cury[j]+ prod_h > stock_h:
                            continue
                        pos_x = self.curx[j]
                        pos_y = self.cury[j]
                        stock_idx = j
                        # cập nhật con trỏ x 
                        self.curx[j] = (self.curx[j] + prod_w)
                        # Chiều cao tối đa của stock thứ j, nếu level mới được mở ra thì cập nhật lại max_height
                        self.max_height[j]=max(self.max_height[j],self.cury[j]+prod_h)
                        break
                    break
            # hết sản phẩm thì reset các mảng lưu trữ về trạng thái ban đầu
            if(numprod==1):
                self.max_height = [0] * 100
                self.curx = [0] * 100
                self.cury = [0] * 100
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        # Student code here
        # You can add more functions if needed
