import numpy as np 
import random 

class Swarm(object):
    """Wrapper class performs optimization between particle objects"""
    def __init__(self, num_particles, function):
        self.numParticles = numParticles
        self.function= function
        self.swarm = [Particle(self.function)]*numParticles  
        self.overBestVal= 0
        self.overBestPos= [0 0]
    def solve(self):
        """
        Calls update() and checkConvergence() until checkConvergence is true
        RETURNS(functionValue, position)
        """
        pass
    def update(self):
        """
        call Particle.compareLocalBest, then compare to global best. Set new global best if necessary
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
    def __init__(self, function, pInit, vInit):
        self.position = [random.uniform(-10,10), random.uniform(-10,10)] # randrange gives 
        self.velocity = [random.uniform(-1,1), random.uniform(-1,1)] # CHANGE: Set a random x and y velocity, with +/-  -> sends out particles in random direction
        self.function = function
        self.functionValue = 0
        self.bestPosition = [0,0]   # check (0,0) first as a possible solution point
        self.bestFuncValue = self.evaluateFunction(0,0)    # calculates function value at (0,0)-- ensures that max is still captured even if max function value <0
    
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
        newValue = self.function(self.position[0], self.position[1])
        if newValue> self.bestFuncValue:
            self.bestFuncValue = newValue
            self.bestPosition= self.position

    def updateVelocity(self, glob):
        """ multiplies weights by best position vectors (local, global) to get new direction
            also somehow calculates and sets new velocity magnitude
            RETUNS(void)
        """
        # Velocity equation parameters that control efficacy and behavior of the PSO, selected arbitrarily by us 
        omega= 1.0     # coefficient for influence of currect velocity
        psi_loc= 1.0   # coefficient for influence of local best
        psi_glob= 0.5  # coefficient for influence of global best 
        #calculates random weights between 0 and 1 (non-inclusive) for influence of individual (local) or social (global) best
        # use maximum() to ensure that random weight is greater than 0 (fullfils the non-inclusive requirement)
        randLocalWeight= np.maximum([0.01, random.random()])
        randGlobalWeight= np.maximum([0.01, random.random()])

        #multiplies weights with best vectors (glob is the Swarm object) and current velocity to get new velocity
        #function below comes from wikipedia.org/wiki/Particle_swarm_optimization
        self.velocity= omega*self.velocity + psi_loc*randLocalWeight*self.bestPosition + psi_glob*randGlobalWeight*glob.overBestPos
        
    def updatePosition(self):
        """ calculates new position based on velocity and time step and updates previous position dictionary w/ new position
            and function value
            RETURNS(void)
        """

        #For this update, a time-step of 1 is assumed ->Change Code if not true
        self.position = [self.position[0] + self.velocity[0], self.position[1]+self.velocity[1]]

    def isFeasible(self):
        """ checks if new position is feasible, getRandomWeights, updateVelocity and updatePosition if not feasible
            RETURNS(boolean)
        """
        pass

        
if __name__ == '__main__':
    class my_function(object):
        def __init__(self):
            """Create function or matrix here"""
            pass
        def evaluate(self,x,y):
            """Takes in two variables, depending on way that the above is created, evaluates
            RETURNS: float of the result"""

            #Current function comes from MATLAB Peaks function
            return 3*(1-x)**2*np.exp(-(x**2)) - (y+1)**2 - 10 * (x/5 - x**3 - y**5) * np.exp(-(x**2) - (y**2)) - (1/3)*np.exp(-((x+1)**2) - y**2)

    PSO = main(8, 't')