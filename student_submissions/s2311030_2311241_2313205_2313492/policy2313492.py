from policy import Policy
import numpy as np

class Policy2313492(Policy):
    def __init__(self):
        # Limit the number of attempts and steps to prevent infinite loops
        self.max_attempts = 1000
        self.max_steps = 4000

    def can_fit(self, stock, position, size):
        """
        Check if a product with a given size can be placed at a specific position in the stock.
        """
        x, y = position
        width, height = size
        if (stock[x][y] != -1 ):
            return False
        # Verify if the product exceeds the boundaries of the stock
        if x + height > stock.shape[0] or y + width > stock.shape[1]:
            return False

        # Ensure the area in the stock is empty and available for placement
        return np.all(stock[x:x + height, y:y + width] == -1)

    def find_best_position(self, stock, product_size):
        """
        Find the first position where the product can be placed in the stock (including trying to rotate the product 90 degrees).
        """
        original_size = product_size
        rotated_size = (product_size[1], product_size[0])  # Size when rotated 90 degrees

        # Try placing the product with both its original and rotated size
        for size in [original_size, rotated_size]:
            height, width = size
            for x in range(stock.shape[0] - height + 1):  # Iterate over rows
                for y in range(stock.shape[1] - width + 1):  # Iterate over columns
                    if self.can_fit(stock, (x, y), (width, height)):
                        return (x, y), size  # Return the valid position and size

        # Return None if no valid position is found
        return None, None

    def bound(self, stocks, products, actions):
        """
        Calculate the lower bound on the number of stocks required to fit all products.
        """
        # Calculate the total area of remaining products
        total_area = sum(p['size'][0] * p['size'][1] * p['quantity'] for p in products)

        # Calculate the total available area across all stocks
        available_area = sum(np.sum(stock != -2) for stock in stocks)

        # Lower bound based on required area vs available area
        lower_bound = np.ceil(total_area / available_area)

        # Count the number of stocks that have already been used
        stock_count = len(set(action['stock_idx'] for action in actions))

        return max(lower_bound, stock_count)

    def sort_stocks_by_area(self, stocks, product_size):
        """
        Sort stocks based on how well they fit a specific product.
        """
        stock_scores = []

        for i, stock in enumerate(stocks):
            stock_height, stock_width = stock.shape
            product_height, product_width = product_size

            # Check if the product can fit into the stock (in either orientation)
            can_fit = (
                (product_height <= stock_height and product_width <= stock_width) or
                (product_width <= stock_height and product_height <= stock_width)
            )

            if can_fit:
                # Calculate the usable area of the stock
                available_area = np.sum(stock != -2)

                # Prioritize stocks with larger areas and better dimension matching
                score = available_area - abs(stock_height - product_height) - abs(stock_width - product_width)
                stock_scores.append((i, score))

        # Sort stocks based on the calculated scores, in descending order
        sorted_stocks = sorted(stock_scores, key=lambda x: x[1], reverse=True)

        # Return the list of sorted stock indices
        return [i for i, _ in sorted_stocks]

    def branch(self, stocks, products, actions):
        """
        Create new search branches by trying to place a product in stocks.
        Priority is given to larger products to maximize space utilization.
        """
        # Sort products by area in descending order
        sorted_products = sorted(
            enumerate(products),
            key=lambda x: x[1]['size'][0] * x[1]['size'][1],
            reverse=True
        )

        best_solution = None
        best_bound = float('inf')  # Start with the highest possible bound

        for product_idx, product in sorted_products:
            if product['quantity'] > 0:  # Only process products with remaining quantity
                sorted_indexes = self.sort_stocks_by_area(stocks, product['size'])

                for stock_idx in sorted_indexes:  # Iterate over stocks in sorted order
                    stock_copy = np.copy(stocks[stock_idx])  # Create a copy of the stock to avoid modifying the original
                    position, size = self.find_best_position(stock_copy, product['size'])

                    if position is not None:
                        # Place the product in the stock
                        x, y = position
                        width, height = size
                        stock_copy[x:x + height, y:y + width] = product_idx

                        # Add the action to the list
                        actions.append({
                            "item_index": product_idx,
                            "stock_idx": stock_idx,
                            "position": position,
                            "size": size
                        })

                        # Decrease the product quantity after placement
                        products[product_idx]['quantity'] -= 1

                        # Calculate the new lower bound
                        lower_bound = self.bound(stocks, products, actions)

                        if lower_bound < best_bound:
                            best_solution = actions[:]
                            best_bound = lower_bound

                        # Revert changes after trying this branch
                        products[product_idx]['quantity'] += 1
                        actions.pop()

        return best_solution

    def get_action(self, observation, info):
        """
        Determine the best action using the Branch and Bound method.
        """
        # Convert the list of stocks and products from the observation
        stocks = [np.array(stock) for stock in observation['stocks']]
        products = [{'size': p['size'], 'quantity': p['quantity']} for p in observation['products']]

        actions = []
        best_solution = self.branch(stocks, products, actions)

        if best_solution:
            return best_solution[0]  # Return the first action in the optimal solution
        else:
            print("Warning: No valid action found")
            return {"item_index": -1, "stock_idx": -1, "position": (-1, -1), "size": (-1, -1)}  # Default action if no valid solution is found