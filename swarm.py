import numpy as np 
import random 

class Main(object):
    """docstring for Main"""
    def __init__(self, num_particles, cost_function):
        self.numParticles = numParticles
        self.cost_function = costFunction
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
    """docstring for Particle"""
    def __init__(self):
        self.velocity = [0,0] # CHANGE: Set a random x and y velocity, with +/-  -> sends out particles in random direction
        self.position = [0,0]
        self.functionValue = 0
        self.previousPositions
    
    def evaluateFunction(self):
        """evaluates the objective (cost) function for the particle's current position
            RETURNS (float) functionValue at current position
        """
        pass
    def compareToLocalBest(self):
        """compares the objective function value to individual particle's best known position
            If better, store position and functionValue as local best; if <=, do nothing
            RETURNS (void)
        """
        pass
    def compareToGlobalBest(self):
        """ compares the objective function value to global best known position
            If better, store position and functionValue as global best; if <= do nothing
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
        """ calculates new position based on velocity and time step
            RETURNS(void)
        """
        pass
    def isFeasible(self):
        """ checks if new position is feasible, getRandomWeights, updateVelocity and updatePosition if not feasible
            RETURNS(boolean)
        """
        pass

        
if __name__ == '__main__':
    PSO = main(8, 't')