import numpy as np 
import random 

class Main(object):
    """Wrapper class performs optimization between particle objects"""
    def __init__(self, num_particles):
        self.numParticles = numParticles
        self.swarm = [Particle()]*numParticles  

    def solve(self):
        """
        Calls update() and checkConvergence() until checkConvergence is true
        RETURNS(functionValue, position)
        """
        pass
    def update(self):
        """
        Sets velocities of Particles in swarm, after one time step updates the position and evaluates the function,
        evaluates the feasibility of the point, sets local best for each Particle, sets the global best position and function value
        RETURNS(void)
        """
        pass


    def checkConvergence(self):
        """ counts difference between global_best and individual particle values below a given tolerance
            if count> threshold, convergence = true
            RETURNS(boolean)
        """
        pass

class Particle(object):
    """Class for creating each of the particles, handles knowledge/optimization for single particle"""
    def __init__(self, function):
        self.velocity = [0,0] # CHANGE: Set a random x and y velocity, with +/-  -> sends out particles in random direction
        self.position = [0,0]
        self.function 
        self.functionValue = 0
        self.previousPositions = {}
        self.bestValue = 0
    
    def evaluateFunction(self):
        """evaluates the objective (cost) function for the particle's current position
            RETURNS (float) functionValue at current position
        """
        self.functionValue = self.function.evaluate(self.position[0], self.position[1])
        return self.functionValue

    def compareToLocalBest(self):
        """compares the objective function value to individual particle's best known position
            If better, store position and functionValue as local best; if <=, do nothing
            RETURNS (void)
        """
        for key in self.previousPositions:
            if self.previousPositions[key] > self.bestValue:
                self.bestValue = self.previousPositions[key]

    def compareToGlobalBest(self):
        """ compares the objective function value to global best known position
            If better, store position and functionValue as global best; if <= store self.bestValue
            as global best
            RETURNS(void)
        """
        pass
    def getRandomWeights(self):
        """ calculates random weights between 0 and 1 for influence of individual (local) or social (global) best
            RETURNS ([float, float])
        """
        pass
    def updateVelocity(self):
        """ multiplies weights by best position vectors (local, global) to get new direction
            also somehow calculates and sets new velocity magnitude
            RETUNS(void)
        """
        pass
    def updatePosition(self):
        """ calculates new position based on velocity and time step and updates previous position dictionary w/ new position
            and function value
            RETURNS(void)
        """

        #For this update, a time-step of 1 is assumed ->Change Code if not true
        self.position = [self.position[0] + self.velocity[0], self.position[1]+self.velocity[1]]
        self.previousPositions[self.position] = self.evaluateFunction()

    def isFeasible(self):
        """ checks if new position is feasible, getRandomWeights, updateVelocity and updatePosition if not feasible
            RETURNS(boolean)
        """
        pass

        
if __name__ == '__main__':
    class my_function(object):
        def __init__(self):
            """Create functionn or matrix here"""
            pass
        def evaluate(self,x,y):
            """Takes in two variables, depending on way that the above is created, evaluates
            RETURNS: float of the result"""

            #Current function comes from MATLAB Peaks function
            return 3*(1-x)**2*np.exp(-(x**2)) - (y+1)**2 - 10 * (x/5 - x**3 - y**5) * np.exp(-(x**2) - (y**2)) - (1/3)*np.exp(-((x+1)**2) - y**2)

    PSO = main(8, 't')