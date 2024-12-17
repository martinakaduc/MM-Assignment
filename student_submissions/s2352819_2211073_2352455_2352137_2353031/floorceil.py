from policy import Policy  # Import the base Policy class from the policy module


class FloorCeil(Policy):
    """A heuristic-based policy for placing products into stocks efficiently."""

    def __init__(self, policy_id=1):
        """
        Initialize the Policytest instance.

        Args:
            policy_id (int): Identifier for the policy. Must be a non-zero value.
        """
        assert policy_id  # Ensure that policy_id is provided and valid
        self.stock_index = 0  # Current stock being processed
        self.numItem = 0  # Total number of items to be placed
        self.prodIndex = 0  # Index of the current product in sortedProd
        self.sortedProd = None  # List of products sorted by height (descending)
        self.rotateIndex = 0  # Index for iterating through sortedRotateProd
        self.currentStrip = 0  # Current strip or layer within the stock (usage unclear without more context)
        self.nextStrip = -1  # Next strip index (initialized to -1, possibly indicating no next strip yet)
        self.floorLevel = 0  # Current floor level in the stock where products are being placed
        self.ceilLevel = -1  # Current ceiling level in the stock (initialized to -1)
        self.floorPhase = True  # Flag indicating whether the algorithm is in the floor placement phase
        self.greedySortProd = []  # List of products sorted by area (width * height) for greedy placement

    def get_action(self, observation, info):
        """
        Determine the next action based on the current observation and additional info.

        Args:
            observation (dict): Current state information, including stocks and products.
            info (dict): Additional information (usage depends on broader context).

        Returns:
            dict: Action dictionary specifying stock index, product size, and position.
        """
        return self.heuristic(observation, info)  # Delegate action decision to the heuristic method

    def heuristic(self, observation, info):
        """
        Heuristic method to decide where and how to place the next product.

        Args:
            observation (dict): Current state information, including stocks and products.
            info (dict): Additional information.

        Returns:
            dict: Action dictionary specifying stock index, product size, and position.
        """
        self.resetEp(observation)  # Initialize or reset the episode state based on the current observation
        stock = observation["stocks"][self.stock_index]  # Get the current stock being processed
        stock_w, stock_h = self._get_stock_size_(stock)  # Retrieve the width and height of the current stock

        # Iterate through the sorted list of products
        while self.prodIndex < len(self.sortedProd):
            prod = self.sortedProd[self.prodIndex]  # Get the current product
            if prod["quantity"] > 0:  # Check if there are remaining quantities of this product to place
                # Skip rotated products that have no remaining quantity
                while self.rotateIndex < len(self.sortedRotateProd) and self.sortedRotateProd[self.rotateIndex][
                    "quantity"] <= 0:
                    self.rotateIndex += 1

                isRotate = False  # Flag to indicate if the product has been rotated
                if self.rotateIndex < len(self.sortedRotateProd):
                    rotatedProd = self.sortedRotateProd[self.rotateIndex]  # Get the rotated product
                    # Check if rotating the product makes it wider than its original height
                    if rotatedProd["size"][0] > prod["size"][1]:
                        prod = rotatedProd  # Use the rotated product instead
                        isRotate = True  # Set the rotation flag

                # Determine the size of the product, rotated if applicable
                prod_size = prod["size"][::-1] if isRotate else prod["size"]
                prod_w, prod_h = prod_size  # Unpack product width and height

                # Check if the product fits within the current stock dimensions
                if stock_w < prod_w or stock_h < prod_h:
                    self.updateIndex(isRotate)  # Update the product or rotated index
                    continue  # Move to the next product

                if self.floorPhase:  # If in the floor placement phase
                    # Check if the product's height fits within the remaining vertical space from the floor level
                    if prod_h > stock_h - self.floorLevel:
                        self.updateIndex(isRotate)  # Update the index if it doesn't fit
                        continue  # Move to the next product

                    # Attempt to place the product starting from the left side of the current floor level
                    for x in range(stock_w - prod_w + 1):
                        if self._can_place_(stock, (x, self.floorLevel), prod_size):
                            if self.ceilLevel < self.floorLevel:
                                self.ceilLevel = self.floorLevel + prod_h - 1  # Update the ceiling level if needed
                            self.numItem -= 1  # Decrement the total number of items to place
                            return {
                                "stock_idx": self.stock_index,
                                "size": prod_size,
                                "position": (x, self.floorLevel)
                            }  # Return the placement action
                else:  # If in the ceiling placement phase
                    # Check if the product's height fits within the available space between floor and ceiling levels
                    if prod_h > self.ceilLevel - self.floorLevel + 1:
                        self.updateIndex(isRotate)  # Update the index if it doesn't fit
                        continue  # Move to the next product

                    # Attempt to place the product starting from the right side of the current ceiling level
                    for x in range(stock_w - 1, prod_w - 1, -1):
                        if self._can_place_(stock, (x - prod_w + 1, self.ceilLevel - prod_h + 1), prod_size):
                            self.numItem -= 1  # Decrement the total number of items to place
                            return {
                                "stock_idx": self.stock_index,
                                "size": prod_size,
                                "position": (x - prod_w + 1, self.ceilLevel - prod_h + 1)
                            }  # Return the placement action

                self.updateIndex(isRotate)  # Update the index if placement was unsuccessful
                continue  # Continue with the next product

            self.prodIndex += 1  # Move to the next product if current product's quantity is exhausted

        # If the stock's height limit is reached, attempt greedy placement of remaining products
        if self.ceilLevel == stock_h - 1 or self.floorLevel == stock_h - 1:
            for prod in self.greedySortProd:  # Iterate through products sorted by area (greedy approach)
                if prod["quantity"] > 0:  # Check if there are remaining quantities of this product to place
                    prod_size = prod["size"]  # Get the product's size
                    prod_w, prod_h = prod_size  # Unpack product width and height

                    # Skip the product if it doesn't fit within the stock dimensions
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Attempt to place the product in its original orientation
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                self.numItem -= 1  # Decrement the total number of items to place
                                return {
                                    "stock_idx": self.stock_index,
                                    "size": prod_size,
                                    "position": (x, y)
                                }  # Return the placement action

                    # Attempt to place the product in its rotated orientation
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]  # Rotate the product size
                                self.numItem -= 1  # Decrement the total number of items to place
                                return {
                                    "stock_idx": self.stock_index,
                                    "size": prod_size,
                                    "position": (x, y)
                                }  # Return the placement action

            self.moveToNextStock()  # Move to the next stock if greedy placement fails
            return self.heuristic(observation, info)  # Recursively call heuristic to continue placement

        # Switch between floor and ceiling phases
        if self.floorPhase:
            self.floorPhase = False  # Switch to ceiling phase
            self.rotateIndex = 0  # Reset the rotation index
            self.prodIndex = 0  # Reset the product index
            if self.ceilLevel < self.floorLevel:
                self.ceilLevel = stock_h - 1  # Update ceiling level if necessary
        else:
            self.rotateIndex = 0  # Reset the rotation index
            self.prodIndex = 0  # Reset the product index
            self.floorPhase = True  # Switch back to floor phase
            self.floorLevel = self.ceilLevel + 1  # Update floor level based on the current ceiling level

        return self.heuristic(observation, info)  # Recursively call heuristic to continue placement

    def resetEp(self, observation):
        """
        Reset the episode state based on the current observation.

        Args:
            observation (dict): Current state information, including stocks and products.
        """
        if self.numItem == 0:  # If no items have been placed yet
            list_prods = observation["products"]  # Get the list of products from the observation
            self.numItem = sum(prod["quantity"] for prod in list_prods)  # Calculate the total number of items to place
            self.prodIndex = 0  # Reset the product index
            self.rotateIndex = 0  # Reset the rotation index
            self.stock_index = 0  # Reset the stock index

            # Sort products by height in descending order for primary placement
            self.sortedProd = sorted(list_prods, key=lambda prod: prod["size"][1], reverse=True)

            # Sort products by width in descending order to handle rotations
            self.sortedRotateProd = sorted(list_prods, key=lambda prod: prod["size"][0], reverse=True)

            # Sort products by area (width * height) in descending order for greedy placement
            self.greedySortProd = sorted(
                self.sortedProd,
                key=lambda prod: prod["size"][1] * prod["size"][0],
                reverse=True
            )

    def moveToNextStock(self):
        """
        Move to the next stock and reset relevant indices and levels.
        """
        self.stock_index += 1  # Increment the stock index to process the next stock
        self.prodIndex = 0  # Reset the product index for the new stock
        self.rotateIndex = 0  # Reset the rotation index for the new stock
        self.floorPhase = True  # Start with the floor phase in the new stock
        self.ceilLevel = -1  # Reset the ceiling level
        self.floorLevel = 0  # Reset the floor level

    def updateIndex(self, isRotate):
        """
        Update the product or rotation index based on whether the product was rotated.

        Args:
            isRotate (bool): Flag indicating if the current product was rotated.
        """
        if isRotate:
            self.rotateIndex += 1  # Move to the next rotated product if rotated
        else:
            self.prodIndex += 1  # Move to the next product if not rotated
