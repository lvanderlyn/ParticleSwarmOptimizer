import numpy as np 
import random 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pylab 
from mpl_toolkits.mplot3d import Axes3D
mapping = cm.jet
class Swarm(object):
    """Wrapper class performs optimization between particle objects"""
    def __init__(self, num_particles, function, disp = False):
        self.numParticles = num_particles
        self.function= function
        self.swarm = [Particle(self.function) for i in range(self.numParticles)] 
        self.overBestVal= -float('Inf')
        self.overBestPos= [0, 0]
        self.disp = disp

    def solve(self):
        """
        Calls update() and checkConvergence() until checkConvergence is true
        RETURNS(position, functionValue)
        """
        iteration = 0
        converged = False
        plt.hold(True)
        fileadd = 0;
        while not converged:
        # for i in range(4):
            self.update()
            converged = self.checkConvergence(iteration)
            iteration = iteration+1
            if self.disp:
                pylab.savefig('peaks%d.jpg' %fileadd)
                fileadd=fileadd+1
        return [self.overBestPos, self.overBestVal]

    def update(self):
        """
        call Particle.compareToLocalBest, then compare to global best. Set new global best,
        Sets velocities of Particles in swarm, after one time step updates the position and evaluates the function,
        evaluates the feasibility of the point
        RETURNS(void)
        """

        for Particle in self.swarm: 
            Particle.compareToLocalBest()

        self.overBestVal= max(self.overBestVal, max([Particle.bestXYZ[2] for Particle in self.swarm]))
        tempPos= [particle.bestXYZ[0:2] for particle in self.swarm if particle.bestXYZ[2]==self.overBestVal]
        if len(tempPos)>0:
            self.overBestPos = tempPos[0] #In the case that multiple particles have the same max value, use position of 1st particle
        print('Global Best value')
        print(self.overBestVal)
        # print(self.overBestPos, self.overBestVal)
        # print("Current Position, Current Velocity, Current value")
        for index in range(len(self.swarm)): 
            z= function.evaluate(X,Y)
            plt.hold(True)
            cs = plt.contour(X, Y, z)
            if self.disp:
                c = mapping(int(255/(index+1)))
                plt.plot(self.swarm[index].position[0], self.swarm[index].position[1], '*', mfc = c, mec = c)    
            self.swarm[index].updateVelocity(self)
            self.swarm[index].updatePosition()
            self.swarm[index].evaluateFunction()

            # print(Particle.position, Particle.velocity[0], Particle.functionValue)

    def checkConvergence(self, iteration):
        """ Looks for all points converging (stdev) and/or the global maximum remain near constant for a certain period of time 
            RETURNS(boolean)
        """
        threshold = abs(0.01*self.overBestVal)
        stdev = np.std(np.array([particle.bestXYZ[2] for particle in self.swarm]))
        return stdev <= threshold or iteration > 200

class Particle(object):
    """Class for creating each of the particles, handles knowledge/optimization for single particle"""
    def __init__(self, function):
        self.position = np.array([random.uniform(-120,120), random.uniform(-120,120)]) # randrange gives 
        self.velocity = np.array([random.uniform(-1,1), random.uniform(-1,1)]) # CHANGE: Set a random x and y velocity, with +/-  -> sends out particles in random direction
        self.function = function
        self.functionValue = 0
        print(self.position)
        # self.bestPosition = [0,0]   # check (0,0) first as a possible solution point
        # self.bestFuncValue = self.evaluateFunction(0,0)    # calculates function value at (0,0)-- ensures that max is still captured even if max function value <0
        self.bestXYZ = np.array([self.position[0],self.position[1],self.evaluateFunction()]) #XYZ list where z is the function value and x and y are the particle's position coordinates
    def evaluateFunction(self):
        """evaluates the objective (cost) function for the particle's current position
            RETURNS (void)
        """
        self.functionValue = self.function.evaluate(self.position[0], self.position[1])
        
    def compareToLocalBest(self):
        """compares the objective function value to individual particle's best known position
            If better, store position and functionValue as local best; if <=, do nothing
            RETURNS (void)
        """
        if self.functionValue> self.bestXYZ[2]:
            self.bestXYZ[2] = self.functionValue
            self.bestXYZ[0:2] = self.position
        self.bestXYZ = np.array(self.bestXYZ)

    def updateVelocity(self, glob):
        """ multiplies weights by best position vectors (local, global) to get new direction
            also somehow calculates and sets new velocity magnitude
            RETUNS(void)
        """
        # Velocity equation parameters that control efficacy and behavior of the PSO, selected arbitrarily by us 
        omega= .30     # coefficient for influence of currect velocity
        psi_loc= 0.40   # coefficient for influence of local best
        psi_glob= 0.30  # coefficient for influence of global best 
        #calculates random weights between 0 and 1 (non-inclusive) for influence of individual (local) or social (global) best
        # use maximum() to ensure that random weight is greater than 0 (fullfils the non-inclusive requirement)
        randLocalWeight= max(0.01, random.random())
        randGlobalWeight= max(0.01, random.random())

        #multiplies weights with best vectors (glob is the Swarm object) and current velocity to get new velocity
        #function below comes from wikipedia.org/wiki/Particle_swarm_optimization
        self.velocity= omega*self.velocity + psi_loc*(self.bestXYZ[0:2] - self.position)*randLocalWeight + psi_glob*(np.array(glob.overBestPos)-self.position)*randGlobalWeight
        
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
            return 3*(np.power((1-x),2)).dot(np.exp(-np.power(x,2))) - np.power((y+1),2) - 10 * (x/5 - np.power(x,3) - np.power(y,5)).dot(np.exp(-np.power(x,2) - np.power(y,2))) - (1/3)*np.exp(-np.power((x+1),2) - np.power(y,2))
            # return -np.power((x-50),2) - np.power(y, 2)-3
    N = 100
    x = np.linspace(-150.0, 150.0, N)
    y = np.linspace(-150.0, 150.0, N)
    function= my_function()
    X, Y = np.meshgrid(x, y)
    Z = function.evaluate(X,Y)
    # PSO = Swarm(15, function, True)
    # PSO.solve()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot_surface(X,Y,Z)
    plt.show()


