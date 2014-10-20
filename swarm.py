import numpy as np 
import random 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pylab 
from mpl_toolkits.mplot3d import Axes3D
mapping = cm.jet
class Swarm(object):
    """Wrapper class performs optimization between particle objects"""
    def __init__(self, num_particles, function, disp = False, latency = 1, reset = None):
        """
        num_particles: controls the number of independent searchers
        function: function object, should have function and constraints methods
        disp: boolean, controls whether images of iterations are saved
        latency: number of time-steps before function value is evaluated
        reset: number of time time-steps before particles are reset to local best positions (must be >0)
        """
        self.numParticles = num_particles
        self.function= function
        self.swarm = [Particle(self.function) for i in range(self.numParticles)] 
        self.overBestVal= -float('Inf')
        self.overBestPos= [0, 0]
        self.disp = disp
        self.bestStreak=0
        self.latency = latency
        self.reset = reset
        self.iteration = 0

    def solve(self):
        """
        Calls update() and checkConvergence() until checkConvergence is true
        RETURNS(position, functionValue)
        """
        converged = False
        plt.hold(True)
        fileadd = 0;
        while not converged:
        # for i in range(4):
            self.update()
            converged = self.checkConvergence(self.iteration)
            self.iteration += 1
            if self.disp:
                pylab.savefig('reest10_%d.jpg' %fileadd)  # If display is turned on, save figure for visualization
                fileadd += 1
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

        self.prevBestVal= self.overBestVal
        self.overBestVal= max(self.overBestVal, max([Particle.bestXYZ[2] for Particle in self.swarm]))
        tempPos= [particle.bestXYZ[0:2] for particle in self.swarm if particle.bestXYZ[2]==self.overBestVal]
        if len(tempPos)>0:
            self.overBestPos = tempPos[0] #In the case that multiple particles have the same max value, use position of 1st particle
        print('Global Best value')
        print(self.overBestVal)
        # print(self.overBestPos, self.overBestVal)
        # print("Current Position, Current Velocity, Current value")
        for index in range(len(self.swarm)): 
            if self.disp:
                c = mapping(int(255/(index+1)))
                plt.plot(self.swarm[index].position[0], self.swarm[index].position[1], '*', mfc = c, mec = c)    
            self.swarm[index].updateVelocity(self, self.latency) #multiplies velocity by #time-steps until update
            if self.reset != None:
                if self.iteration % self.reset == 0:
                    self.swarm[index].position = self.swarm[index].bestXYZ[0:2]
                else:
                    self.swarm[index].updatePosition()
            else:
                self.swarm[index].updatePosition()
            self.swarm[index].isFeasible()
            self.swarm[index].evaluateFunction()

            # print(Particle.position, Particle.velocity[0], Particle.functionValue)

    def checkConvergence(self, iteration):
        """ Looks for all points converging (stdev) and/or the global maximum remain near constant for a certain period of time 
            RETURNS(boolean)
        """
        threshold = abs(0.05*self.overBestVal)
        stdev = np.std(np.array([particle.bestXYZ[2] for particle in self.swarm]))
        if self.overBestVal==self.prevBestVal:
            self.bestStreak+=1
        else:
            self.bestStreak=0
        if stdev<=threshold:
            print('Converged. Stdev below threshold')
        elif self.bestStreak>=50:
            print('Converged. Best streak length greater than threshold')
        elif iteration>=800:
            print('Did not converge.')
        return stdev <= threshold or self.bestStreak>=50 or iteration>=800

class Particle(object):
    """Class for creating each of the particles, handles knowledge/optimization for single particle"""
    def __init__(self, function):
        self.position = np.array([random.uniform(-50,50), random.uniform(-50,50)]) 
        self.velocity = np.array([random.uniform(-1,1), random.uniform(-1,1)]) 
        self.function = function
        self.functionValue = 0      
        self.bestXYZ = np.array([self.position[0],self.position[1],self.evaluateFunction()]) #XYZ list where z is the function value and x and y are the particle's position coordinates
    
    def evaluateFunction(self):
        """evaluates the objective (cost) function for the particle's current position
            RETURNS (void)
        """
        self.functionValue = np.round(self.function.evaluate(self.position[0], self.position[1]), 2)
        
    def compareToLocalBest(self):
        """compares the objective function value to individual particle's best known position
            If better, store position and functionValue as local best; if <=, do nothing
            RETURNS (void)
        """
        if self.functionValue> self.bestXYZ[2]:
            self.bestXYZ[2] = self.functionValue
            self.bestXYZ[0:2] = self.position
        self.bestXYZ = np.array(self.bestXYZ)

    def updateVelocity(self, glob, latency):
        """ multiplies weights by best position vectors (local, global) to get new direction
            also somehow calculates and sets new velocity magnitude
            RETUNS(void)
        """
        # Velocity equation parameters that control efficacy and behavior of the PSO, selected arbitrarily by us 
        omega= .40     # coefficient for influence of currect velocity
        psi_loc= 0.30   # coefficient for influence of local best
        psi_glob= 0.30  # coefficient for influence of global best 
        #calculates random weights between 0 and 1 (non-inclusive) for influence of individual (local) or social (global) best
        # use maximum() to ensure that random weight is greater than 0 (fullfils the non-inclusive requirement)
        randLocalWeight= .01*random.randrange(-100,100,1)
        randGlobalWeight= .01*random.randrange(-100,100,1)
        #randVel= max(.01, random.random())
        #multiplies weights with best vectors (glob is the Swarm object) and current velocity to get new velocity
        #function below comes from wikipedia.org/wiki/Particle_swarm_optimization
        self.velocity= (omega*self.velocity + psi_loc*(self.bestXYZ[0:2] - self.position)*randLocalWeight + psi_glob*(np.array(glob.overBestPos)-self.position)*randGlobalWeight)*latency
        # latency multiplies velocity by #time-steps until update
    def updatePosition(self):
        """ calculates new position based on velocity and time step and updates previous position dictionary w/ new position
            and function value
            RETURNS(void)
        """

        #For this update, a time-step of 1 is assumed ->Change Code if not true
        self.position = [self.position[0] + self.velocity[0], self.position[1]+self.velocity[1]]


    def isFeasible(self):
        """ checks if new position is feasible, getRandomWeights, updateVelocity and updatePosition if not feasible
            RETURNS(void)
        """
        if self.function.constraints(self.position[0],self.position[1]) == False:
            self.position = np.array([random.uniform(-50,50), random.uniform(-50,50)]) 
            self.velocity = np.array([random.uniform(-1,1), random.uniform(-1,1)]) 

        
if __name__ == '__main__':
    class my_function(object):
        def __init__(self,scale):
            """Create function or matrix here"""
            self.scale = scale
            
        def constraints(self, x,y):
            """RETURNS: boolean for whether the proposed position value is feasible (true if yes, false if not)"""
            if  abs(x) >= self.scale or abs(y) >= self.scale:
                return False
            return True
        def evaluate(self,x,y):
            """Takes in two variables, depending on way that the above is created, evaluates
            RETURNS: float of the result"""

            #Current function comes from MATLAB Peaks function
            # return np.multiply(3*np.power((1-x), 2), np.exp(-np.power(x,2) - np.power((y+1), 2))) - np.multiply(10 * (x/5.0 - np.power(x,3) - np.power(y,5)), np.exp(-np.power(x,2)-np.power(y,2)))#- np.exp(-np.power(x+1,2)-np.power(y,2))/3.0
            # return -np.power((x-50),2) - np.power(y, 2)-3
            return 5- (np.multiply(np.multiply(np.sin(x), np.sin(y)), np.power(x,2)) + np.power(y,2))
    N = 10000
    x = np.linspace(-100.0, 100.0, N)
    y = np.linspace(-100.0, 100.0, N)
    function= my_function(100)
    X, Y = np.meshgrid(x,y)
    z = function.evaluate(X,Y)
    # plt.contour(X, Y, z)
    # PSO = Swarm(25, function, disp = True, latency = 1, reset = 12)
    # PSO.solve()
    print np.amax(z)
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection = '3d')
    # ax.plot_surface(X,Y,z)
    # plt.show()


