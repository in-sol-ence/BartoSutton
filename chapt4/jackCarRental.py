import numpy as np

class JackCarRental:
    def __init__(self, location1_cars, location2_cars, version=1):
        self.poisson_lambda_rent1 = 3
        self.poisson_lambda_rent2 = 4
        self.poisson_lambda_return1 = 3
        self.poisson_lambda_return2 = 2
        self.rentRevenue = 10
        self.moveCost = -5
        self.max_cars = 20
        self.max_move = 5
        self.move_cost = 2
        self.location1_cars = location1_cars
        self.location2_cars = location2_cars
        self.cash = 0
    
    def action(self, move_cars):
        ''' Move cars between locations. Positive values move cars from location 1 to location 2,
            negative values move cars from location 2 to location 1. '''
        if abs(move_cars) > self.max_move:
            raise ValueError("Cannot move more than 5 cars in one night.")
        if move_cars > 0:
            if move_cars > self.location1_cars:
                raise ValueError("Not enough cars at location 1 to move.")
        if move_cars == 0:
            return
        else:
            if -move_cars > self.location2_cars:
                raise ValueError("Not enough cars at location 2 to move.")
        
        self.location1_cars -= move_cars
        self.location2_cars += move_cars

        cost = abs(move_cars) * self.move_cost
        self.cash -= cost

        return cost

    def day(self):
        ''' Simulate a day of rentals and returns. '''
        # Rentals
        rent1 = np.random.poisson(self.poisson_lambda_rent1)
        rent2 = np.random.poisson(self.poisson_lambda_rent2)
        actual_rent1 = min(rent1, self.location1_cars)
        actual_rent2 = min(rent2, self.location2_cars)
        self.location1_cars -= actual_rent1
        self.location2_cars -= actual_rent2
        revenue = (actual_rent1 + actual_rent2) * 10 

        # Returns
        # Assuming returns happen after rentals
        return1 = np.random.poisson(self.poisson_lambda_return1)
        return2 = np.random.poisson(self.poisson_lambda_return2)
        self.location1_cars = min(self.location1_cars + return1, self.max_cars)
        self.location2_cars = min(self.location2_cars + return2, self.max_cars)

        self.cash += revenue
        return revenue

    def get_state(self):
        return (self.location1_cars, self.location2_cars, self.cash)
        
    