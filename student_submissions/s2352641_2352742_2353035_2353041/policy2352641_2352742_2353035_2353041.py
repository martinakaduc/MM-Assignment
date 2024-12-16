from policy import Policy
# my includes
from abc import ABC, abstractmethod
import numpy as np
# Rectangle class, contains which product is in which bin at which position
class Rectangle:
    def __init__(self, width, height, rid):
        self.width = width
        self.height = height
        self.original_width = width
        self.original_height = height
        self.rid = rid  # Rectangle ID
        self.x = None
        self.y = None
        self.rotated = False
        self.bin_index = None  # The index of the bin where the rectangle is placed

class MaxRects(ABC):
    def __init__(self, width, height):
        self.bin_width = width
        self.bin_height = height
        self.free_rectangles = []
        self.placed_rectangles = []
        self.fill_percentage = 0.0
        # Initialize with the bin size as the first free rectangle
        self.free_rectangles.append({'x': 0, 'y': 0, 'width': width, 'height': height})

    def is_contained_in(self, rect_a, rect_b):
        return (rect_a['x'] >= rect_b['x'] and
                rect_a['y'] >= rect_b['y'] and
                rect_a['x'] + rect_a['width'] <= rect_b['x'] + rect_b['width'] and
                rect_a['y'] + rect_a['height'] <= rect_b['y'] + rect_b['height'])

    def split_free_rectangle(self, free_rect, used_rect):
        new_rects = []

        # Left
        if used_rect['x'] > free_rect['x']:
            new_rect = {
                'x': free_rect['x'],
                'y': free_rect['y'],
                'width': used_rect['x'] - free_rect['x'],
                'height': free_rect['height']
            }
            new_rects.append(new_rect)
        # Right
        if used_rect['x'] + used_rect['width'] < free_rect['x'] + free_rect['width']:
            new_rect = {
                'x': used_rect['x'] + used_rect['width'],
                'y': free_rect['y'],
                'width': (free_rect['x'] + free_rect['width']) - (used_rect['x'] + used_rect['width']),
                'height': free_rect['height']
            }
            new_rects.append(new_rect)
        # Bottom
        if used_rect['y'] > free_rect['y']:
            new_rect = {
                'x': free_rect['x'],
                'y': free_rect['y'],
                'width': free_rect['width'],
                'height': used_rect['y'] - free_rect['y']
            }
            new_rects.append(new_rect)
        # Top
        if used_rect['y'] + used_rect['height'] < free_rect['y'] + free_rect['height']:
            new_rect = {
                'x': free_rect['x'],
                'y': used_rect['y'] + used_rect['height'],
                'width': free_rect['width'],
                'height': (free_rect['y'] + free_rect['height']) - (used_rect['y'] + used_rect['height'])
            }
            new_rects.append(new_rect)

        return new_rects

    def place_rectangle(self, rect, best_node):
        rect.x = best_node['x']
        rect.y = best_node['y']
        rect.rotated = best_node.get('rotated', False)
        if rect.rotated:
            rect.width = rect.original_height
            rect.height = rect.original_width
        else:
            rect.width = rect.original_width
            rect.height = rect.original_height
        self.placed_rectangles.append(rect)

        i = 0
        while i < len(self.free_rectangles):
            free_rect = self.free_rectangles[i]
            if self.intersect(free_rect, best_node):
                new_rects = self.split_free_rectangle(free_rect, best_node)
                self.free_rectangles.pop(i)
                self.free_rectangles.extend(new_rects)
                i -= 1
            i += 1

        self.prune_free_list()

    def intersect(self, rect_a, rect_b):
        return not (rect_a['x'] >= rect_b['x'] + rect_b['width'] or
                    rect_a['x'] + rect_a['width'] <= rect_b['x'] or
                    rect_a['y'] >= rect_b['y'] + rect_b['height'] or
                    rect_a['y'] + rect_a['height'] <= rect_b['y'])

    def prune_free_list(self):
        i = 0
        while i < len(self.free_rectangles):
            j = i + 1
            while j < len(self.free_rectangles):
                rect_i = self.free_rectangles[i]
                rect_j = self.free_rectangles[j]
                if self.is_contained_in(rect_i, rect_j):
                    self.free_rectangles.pop(i)
                    i -= 1
                    break
                if self.is_contained_in(rect_j, rect_i):
                    self.free_rectangles.pop(j)
                    j -= 1
                j += 1
            i += 1

    @abstractmethod
    def score(self, original_width, original_height, free_rect):
        pass

    def find_pos(self, rect):
        best_score = (float('inf'), float('inf'))
        best_node = None
        for free_rect in self.free_rectangles:
            # No rotation fit
            if rect.original_width <= free_rect['width'] and rect.original_height <= free_rect['height']:
                score = self.score(rect.original_width, rect.original_height, free_rect)
                if score < best_score:
                    best_node = {
                        'x': free_rect['x'],
                        'y': free_rect['y'],
                        'width': rect.original_width,
                        'height': rect.original_height,
                        'rotated': False
                    }
                    best_score = score
            # Rotated fit
            if rect.original_height <= free_rect['width'] and rect.original_width <= free_rect['height']:
                score = self.score(rect.original_height, rect.original_width, free_rect)
                if score < best_score:
                    best_node = {
                        'x': free_rect['x'],
                        'y': free_rect['y'],
                        'width': rect.original_height,
                        'height': rect.original_width,
                        'rotated': True
                    }
                    best_score = score
        return best_node

    def insert(self, rect):
        best_node = self.find_pos(rect)
        if best_node:
            self.place_rectangle(rect, best_node)
            self.fill_percentage += (rect.width * rect.height) / (self.bin_width * self.bin_height)
            return True

        return False

class MaxRects_BSSF(MaxRects):
    def score(self, width, height, free_rect):
        leftover_h = abs(free_rect['width'] - width)
        leftover_v = abs(free_rect['height'] - height)
        short_side_fit = min(leftover_h, leftover_v)
        long_side_fit = max(leftover_h, leftover_v)
        return (short_side_fit, long_side_fit)

class MaxRects_BL(MaxRects):
    def score(self, width, height, free_rect):
        return (height, width)

mr_vars = [MaxRects_BSSF, MaxRects_BL]
sort_modes = [(True, True)]

# Greedy part
def pack_rectangles(rectangles, idx_sheets, heuristic_type_idx=1, sort_mode_idx=3, verbose=False):
    """
    pack_rectangles function
    Greedy
    
    Arguments:
    rectangles: list of Rectangle objects
    idx_sheets: enumerated sheets
    
    Output:
    bin_packs: list of bins (either MaxRects_BSSF, SkylineBinPack) with placed_rectangles attribute containing placed rectangles
    """
    bin_packs = []
    sort_mode = sort_modes[sort_mode_idx]
    # FFD-like algorithm, place large products first
    sorted_rectangles = sorted(rectangles, key=lambda r: r.width*r.height, reverse=sort_mode[0])
    # If product category is large (diverse) enough, using larger bins first may benefit
    sorted_bins = sorted(idx_sheets, key=lambda b: b[0]*b[1],reverse=sort_mode[1])

    # Traverse all rectangles
    for rect in sorted_rectangles:
        placed = False

        # Check for used bin packs, if can fit then use
        for bin_index, bin_pack in bin_packs:
            if bin_pack.insert(rect):
                rect.bin_index = bin_index 
                placed = True
                break

        # Else, use new bin
        if not placed:
            for w, h, idx in sorted_bins:
                if any(bp[0] == idx for bp in bin_packs):
                    continue
                
                can_fit = (rect.original_width <= w and rect.original_height <= h) or \
                          (rect.original_height <= w and rect.original_width <= h)
                if not can_fit:
                    continue
                
                # Create a new bin pack for this bin
                bin_pack = mr_vars[heuristic_type_idx](w, h)
                # insert success
                if bin_pack.insert(rect):
                    # Insert newly used bin and sort for next placement
                    sorted_bins.remove((w, h, idx))
                    bin_packs.append((idx, bin_pack))
                    # BFD sort
                    bin_packs = sorted(bin_packs, key=lambda r: r[1].fill_percentage, reverse=True)
                    # bin_packs = sorted(bin_packs, key=lambda r: r[1].bin_width + r[1].bin_height, reverse=True)
                    

                    rect.bin_index = idx
                    placed = True
                    break
                else:
                    pass

            # this assumes all bins have been iterated through, if fail then 100% no solution
            if not placed:
                return None

    # if verbose:
    #     print("All products have been successfully packed.")
    return bin_packs

def heuristic_2d_csp(given_rectangles, sheets, heuristic_type_idx=1, sort_mode_idx=3, verbose=False):
    rectangles = []
    for idx, rect in enumerate(given_rectangles):
        rectangles.append(Rectangle(rect[0], rect[1], idx))
    idx_sheets = [(w, h, idx) for idx, (w, h) in enumerate(sheets)]
    # Pack rectangles
    bin_packs = pack_rectangles(rectangles, idx_sheets, heuristic_type_idx, sort_mode_idx, verbose)
    if bin_packs is None:
        return None
    # Calculate fill percentage
    bin_packs_dict = {bin_index: bin_pack for bin_index, bin_pack in bin_packs}
    result = {}
    for k in range(len(sheets)):
        if k in bin_packs_dict:
            bin_pack = bin_packs_dict[k]
            for rect in bin_pack.placed_rectangles:
                result[rect.rid] = {
                    'sheet': k,
                    'x': rect.x,
                    'y': rect.y,
                    'rect': (rect.width, rect.height)
                }
    return result

class Policy2352641_2352742_2353035_2353041(Policy):
    """
    Policy implementation (port) from full solution to single solution giver.
    
    Logic:
    Since it is ruled that an Observation is done IFF all products are fulfilled,
    An Observation is unique when result list is depleted.
    If new observation, single solve it then return each product placement sequentially.
    """
    
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.idxVar = 0
        # Student code here
        if policy_id == 1:
            # BSSF - TT
            self.idxVar = 0
            pass
        elif policy_id == 2:
            # BL - TT
            self.idxVar = 1
            pass
        # Save policy id
        self.policy_id = policy_id
        self.result = []
        pass

    # Student code here
    # You can add more functions if needed (no need!)
    def get_action(self, observation, info):
        # Student code here
        if len(self.result) > 0:
            # Protector
            att = self.result.pop(0)
            return {"stock_idx": att['sheet'], "size": np.array(att['rect']), "position": (att['x'], att['y'])}
        # new res
        else:
            self.__init__(self.policy_id)
            products = []
            stocks = []
            # Convert to traditional array of tuples
            for prods in observation['products']:
                size = prods['size']
                quantity = prods['quantity']
                products.extend([tuple(int(x) for x in size)] * quantity)
            # Reconvert weird np array layout stocks 
            for stock in observation['stocks']:
                stocks.append(tuple(int(x) for x in self._get_stock_size_(stock)))
            # Info Printing
            self.result = heuristic_2d_csp(products, stocks, self.idxVar, 0, False)
            # Make into list to put gradually
            self.result = list(self.result.values())
            att = self.result.pop(0)
            return {"stock_idx": att['sheet'], "size": (att['rect']), "position": (att['x'], att['y'])}
        pass
