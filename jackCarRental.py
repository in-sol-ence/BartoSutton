class JackCarRental:
    def __init__(self, location1_cars, location2_cars):
        self.location1_cars = location1_cars
        self.location2_cars = location2_cars
        self.max_cars = 20
        self.rental_reward = 10
        self.move_cost = 2
        self.poisson_lambda_rent1 = 3
        self.poisson_lambda_rent2 = 4
        self.poisson_lambda_return1 = 3
        self.poisson_lambda_return2 = 2
    
    