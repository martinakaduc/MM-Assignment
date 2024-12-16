import numpy as np

class ProductList:
    def __init__(self, products):
        self.product_list = [];
        
        for product in products:
            # rearrange the product list and convert the size to tuple
            self.product_list.append({"size": (product["size"][0].item(), product["size"][1].item()), "quantity": product["quantity"]});
        
    def get_product(self, index):
        return self.product_list[index];
    
    def get_quantity(self, index):
        return self.product_list[index]["quantity"];
    
    def get_size(self, index):
        return self.product_list[index]["size"];
    
    # Return a mapped product list with the index and the product, default key is the size of the product
    def get_mapped_product_list(self, key=lambda x: (x[1]["size"][0], x[1]["size"][1]), reverse=True):
        mapped_product_list = [{"size": (product["size"][0], product["size"][1]), "quantity": product["quantity"]} for product in self.product_list];
        return sorted(enumerate(mapped_product_list), key=key, reverse=reverse);
    
    # Return a deep copy of the product list
    def get_deep_copy(self, product_list):
        return [(index, {"size": (product["size"][0], product["size"][1]), "quantity": product["quantity"]}) for index, product in product_list]
    
    def get_product_list(self):
        return self.product_list;
    
    # Add the quantity of the product at index by number
    def add_quantity(self, index, number=1):
        self.product_list[index]["quantity"] += number;
        
    # Remove the quantity of the product at index by number
    def remove_quantity(self, index, number=1):
        self.product_list[index]["quantity"] -= number;