import numpy as np;

class Stock:
    # Init new stock type
    def __init__(self, stock, index, product_list, parent=None, offset=(0, 0), cut_axis=-1):
        self.stock = stock.copy();
        if parent is None:
            self.stock = self.stock.transpose();
        self.index = index;
        self.product_list = product_list;
        self.stock_h = np.sum(np.any(self.stock != -2, axis=1)).item();
        self.stock_w = np.sum(np.any(self.stock != -2, axis=0)).item();
        self.stock_size = tuple([self.stock_h, self.stock_w]);
        self.offset = offset;
        self.cut_axis = cut_axis;
        self.parent = parent;
    
    def get_data(self):
        return self.stock;
    
    def get_stock_size(self):
        return self.stock_size;
    
    def get_stock_wh(self):
        return self.stock_w, self.stock_h;
    
    def get_offset(self):
        return self.offset;
    
    def get_stock_parent(self):
        return self.parent;
    
    def get_stock_index(self):
        return self.index;
    
    def get_cut_axis(self):
        return self.cut_axis;
    
    def can_place(self, position, product):
        pos_x, pos_y = position;
        prod_w, prod_h = product["size"];
        
        return np.all(self.stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1);
    
    def is_possible_cut(self, cut_position, cut_axis):
        return (cut_position, cut_axis) in self.get_possible_cuts();
    
    # Calculate all possible cuts of stock self and calculate the perfect fits of products
    def get_possible_cuts(self):
        possible_cuts = [];
        product_cut_indexes = [];
        perfect_fits = [];
        
        # Get all sorted products that can be placed in the stock
        for index, product in self.product_list.get_mapped_product_list(lambda x: (x[1]["size"][0] * x[1]["size"][1])):
            # Skip if product is run out
            if product["quantity"] <= 0: continue;
            
            product_w, product_h = product["size"];
            stock_w, stock_h = self.get_stock_wh();

            if product_w > stock_w or product_h > stock_h: continue;
            
            # If there is a perfect fit product, add to perfect fits and there is no need to check for possible cuts
            if product_w == stock_w and product_h == stock_h:
                perfect_fits.append(index);
                break;
            
            # if product width < stock width then the possible cut is vertical cut
            if product_w < stock_w:
                possible_cuts.append((product_w, 1));
                product_cut_indexes.append(index);
                
            # if product height < stock height then the possible cut is horizontal cut
            if product_h < stock_h:
                possible_cuts.append((product_h, 0));
                product_cut_indexes.append(index);
        
        return possible_cuts, product_cut_indexes, perfect_fits;
    
    # Create two new stocks by cutting the stock at the given position and axis
    def create_child(self, cut_info):
        cut_position, cut_axis = cut_info;
        
        stock_w, stock_h = self.get_stock_wh();
        
        # Create two new stocks
        stock1 = self.stock.copy();
        stock2 = self.stock.copy();
        
        stock1[:, :] = -2;
        stock2[:, :] = -2;
        
        # Set the initial offset of the new stocks
        stock1_offset = self.offset;
        stock2_offset = self.offset;
        
        # reassign the stock values as well as the offset values
        if cut_axis == 0:
            stock1[:cut_position, :stock_w] = -1;
            stock2[:stock_h - cut_position, :stock_w] = -1;
            stock2_offset = (stock2_offset[0], stock2_offset[1] + cut_position);
        elif cut_axis == 1:
            stock1[:stock_h, :cut_position] = -1;
            stock2[:stock_h, :stock_w - cut_position] = -1;
            stock2_offset = (stock2_offset[0] + cut_position, stock2_offset[1]);
        
        return Stock(stock1, -1, self.product_list, self, stock1_offset, cut_axis), Stock(stock2, -1, self.product_list, self, stock2_offset, cut_axis);
    
    def get_product_list(self):
        return self.product_list;
    
    def get_product_size(self, product_index):
        return self.product_list.get_size(product_index);
    
    def get_product_wh(self, product_index):
        return self.product_list.get_wh(product_index);