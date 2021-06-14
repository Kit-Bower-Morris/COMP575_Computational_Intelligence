#Kit Bower-Morris, 201532917

import numpy as np
import matplotlib.pyplot as plt


"""
Perceptron Class. 
"""
class Perceptron():
    """
    init function. 
    no_of_inputs = 4. - Number of varibles in data set. 
    threshold = 20. - Number of iterations.
    weights - start as all zeros [0,0,0,0]
    bias - starts as 0
    """
    def __init__(self, no_of_inputs, epochs): #set number of inputs and epochs
        self.epochs = epochs
        self.weights = np.zeros(no_of_inputs)
        self.bias = 0

    """
    train fuction.
    Follow perceptron algorithm. 
    """    
    def train(self, training_inputs, labels):
        print(f"Initial Weights: {self.weights}")
        for i in range(self.epochs):
            training_inputs, labels = self.shuffle(training_inputs, labels) #Shuffle order of input data.
            for inputs, label in zip(training_inputs, labels):
                a = np.dot(inputs, self.weights) + self.bias #Activation function.
                if((label*a) <= 0): #Checks the sign of the actication function. If it not the same as label, weight is adjusted. 
                    self.weights += label * inputs #Update formula.
                    self.bias += label
            print(f"Iteration: {i+1} --> Weights: {self.weights}")
        return self.bias, self.weights

    """
    shuffle function. 
    Takes in data and corrisponding labels. 
    shuffles the order, keeping the data labels pair together. 
    """ 
    def shuffle(self, data, labels):
        assert len(data) == len(labels)
        shuffled = np.random.permutation(len(data))
        return data[shuffled], labels[shuffled]

"""
Multi-Layer Perceptron Class. 
"""
class MLP():

    def __init__(self, structure, weights = [], activationType = "sigmoid"):
        """
        Initialises the Multi-layered Perceptron with the given structure. 
        """
        self.weights = []
        self.activationType = activationType
        self.activations = []
        self.derivatives = []
        self.delta = []
        self.bias = []

        #Either randomises weights or uses starting weights that have been supplied. 
        for i in range(len(structure)-1):
            if len(weights) != 0:
                w = np.zeros((structure[i], structure[i + 1]))
                for j in range(structure[i]):
                    w[j, :] = weights[(j * structure[i + 1]) : (j * structure[i + 1]) + structure[i + 1]]
            else:
                w = np.random.rand(structure[i], structure[i + 1])
            self.weights.append(w)
        
        #Initialises activations array, delta array and bias array. 
        for i in range(len(structure)):
            a = np.zeros(structure[i])
            self.activations.append(a)
            self.delta.append(a)
            self.bias.append(a)
            
        
        #Initialises derivatives array.
        for i in range(len(structure) - 1):
            d = np.zeros((structure[i], structure[i + 1]))
            self.derivatives.append(d)
            
        return

    # Forward pass. 
    # Sets data as starting activations. 
    # Using current weights*data + bias for each node. 
    # Records activations for each node. 
    def forwardPass(self, data):
        self.activations[0] = data
        for i, weight in enumerate(self.weights):
            dw = np.dot(data, weight) +self.bias[i+1]
            data = self.activate(dw)
            self.activations[i+1] = data
        return data

    # Backward Pass. 
    # Gets activation for previous layer.
    # Finds delta.
    # Uses next error for next layer. .
    def backwardPass(self, error):
        for i in reversed(range(len(self.derivatives))):
            previousActivations = self.activations[i+1]
            currentActivations = self.activations[i]
            self.delta[i] = error * self.activateDerivative(previousActivations)
            reshapedD = self.delta[i].reshape(self.delta[i].shape[0], -1).T
            currentActivations = currentActivations.reshape(currentActivations.shape[0],-1)
            self.derivatives[i] = np.dot(currentActivations, reshapedD)
            error = np.dot(self.delta[i], self.weights[i].T)
        return

    # Updates weights and bias.
    def gradD(self, learningRate=1):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate
        for i in range(len(self.weights)):
            bias = self.bias[i+1]
            bias += learningRate * self.delta[i]
    
        return

    #Applies selected activation function. 
    def activate(self, x):
        if self.activationType == "sigmoid":
            y = 1.0 / (1 + np.exp(-x))
            return y
        elif self.activationType == "tanh":
            return np.tanh(x)

    #Finds the derivative of the activation function. 
    def activateDerivative(self, x):
        if self.activationType == "sigmoid":
            return x * (1.0 - x)
        elif self.activationType == "tanh":
            return 1 - np.tanh(x)**2

    #Train function. 
    #Shuffles data.
    #Runs forward and backward passes for each data point, updating weights and bias after each iteration. 
    #Records errors for each epoch.
    def train(self, data, labels, epochs, eta):
        errors = np.zeros(epochs)

        for i in range(epochs):
            totalErrors = 0
            data, labels = self.shuffle(data, labels)

            for j, d in enumerate(data):
                label = labels[j]
                output = self.forwardPass(d)
                error = label - output
                totalErrors += np.average((error) ** 2)
                self.backwardPass(error)
                self.gradD(eta)
            
            errors[i] = totalErrors
            
            if i % min(100, int(epochs / 10)) == 0:
                print(f"Iteration: {i}\nCurrent Error {totalErrors/len(data)}")

        print(f"\nFinal Error: {totalErrors/len(data)}")
        return errors     

    #Shuffle Function. 
    def shuffle(self, data, labels):
        assert len(data) == len(labels)
        shuffled = np.random.permutation(len(data))
        return data[shuffled], labels[shuffled]       
        
"""
Enviroment Class. 
"""      
class Enviroment():

    #Init.
    #Initilizes nn structure, activation type, the data and the labels, number of iterations and eta. 
    def __init__(self, structure, activationType, data, labels, iterations, eta):
        self.structure = structure
        self.activationType = activationType
        self.data = data
        self.labels = labels
        self.iterations = iterations
        self.eta = eta
        self.genes = 0
        for i in range(len(self.structure) - 1):
            self.genes += self.structure[i] * self.structure[i+1]
        return

    #Load functition for Single layer perceptron. 
    #Loads iris data for classification problem. 
    def load_data(self):
        from sklearn import datasets        
        self.load = True
        self.iris = datasets.load_iris()
        #divide into features and target variables
        self.perData = self.iris.data[:100, :3]
        label = self.iris.target[:100]
        for i in range(len(label)):
            if label[i] == 0:
                label[i] = -1
        self.perLabel = label
        return 

    #Install Function for Single layer perceptron. 
    #sets custom data to the right format. 
    def install_data(self, data, label):
        self.load = False
        self.perData = data
        for i in range(len(label)):
            if label[i] == 0:
                label[i] = -1
        self.perLabel = label
        return


    #Run Perceptron function. 
    #Creates 3D graph of data with decision boundary. 
    def runPerceptron(self):
        from mpl_toolkits.mplot3d import Axes3D
        self.per = Perceptron((len(self.perData[0])), 20)
        bias, weights = self.per.train(self.perData, self.perLabel)
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.perData[:, 0], self.perData[:, 1], self.perData[:, 2], c=self.perLabel, cmap=plt.cm.Set1, edgecolor='k')
        if self.load:
            ax.set_xlabel(self.iris.feature_names[0])
            ax.set_ylabel(self.iris.feature_names[1])
            ax.set_zlabel(self.iris.feature_names[2])
        z = lambda x,y: (-bias-weights[0]*x-weights[1]*y) / weights[2]
        tmp = np.linspace(0,7,30)
        x,y = np.meshgrid(tmp,tmp)
        ax.plot_surface(x, y, z(x,y))
        plt.show()
 
        return


    #Run Back Propergation function.
    #calls train MLP function. 
    #Graphs Mean Square Error. 
    def runBackProp(self):

        self.mlp = MLP(self.structure, activationType=self.activationType)
        errors = self.mlp.train(self.data, self.labels, self.iterations, self.eta)
        plt.plot(errors/len(self.data))
        plt.xlabel("Iterations")
        plt.ylabel("Mean Square Error")
        plt.show()
        return self.mlp

    # Test Function. 
    # Tests current best weights for self.mlp with test data.
    # Prints expected value and what the nn predicted. 
    # Prints the error between these two. 
    def test(self, data, label):
        outputs = []
        for i in data:
            output = self.mlp.forwardPass(i.flatten())
            outputs.append(output)
        print(f"\nEXPECTED: {label} --> PREDICTED: {np.round_(outputs, 5)}")
        print(f"Error: {(label-np.round_(outputs, 5))}\n")
        return


    #Run Genetic Algorithmn function. 
    #Imports Continuous Gen AI Solver. 
    #Calls fittness function as a negative. 
    #Saves best weight.
    def runGA(self):
        from geneal.genetic_algorithms import ContinuousGenAlgSolver

        def fitness(X):
            return-self.fittness(X)
        
        solver = ContinuousGenAlgSolver(
            n_genes=self.genes,
            fitness_function=fitness,
            pop_size=50, # population size (number of individuals)
            max_gen=self.iterations, # maximum number of generations
            mutation_rate=0.01, # mutation rate to apply to the population
            selection_rate=self.eta, # percentage of the population to select for mating
            selection_strategy="roulette_wheel", # strategy to use for selection.
            problem_type=float, # Defines the possible values as float numbers
            variables_limits=(-2, 2)
        )
        solver.solve()
        best = solver.best_individual_
        self.mlp = MLP(self.structure, weights = best, activationType=self.activationType)
        return self.mlp

    #Fittness function used by runGA and runPSO
    #Calls MLP with current weights for current population or particle. 
    #Saves mean square error from these current weights. 
    def fittness(self, X):
        mlp = MLP(self.structure, weights = X, activationType=self.activationType)
        outputs = mlp.forwardPass(self.data)
        outputs = np.average((self.labels - outputs) ** 2)
        return outputs

    #Run PSO function. 
    #Installs pyswarms. 
    #Finds fittness for all particles.
    #Saves best.
    def runPSO(self):
        import pyswarms as ps
        from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
        
        # Mass fitness for all particles
        def fit(X):
            n_particles = X.shape[0]
            j = [self.fittness(X[i]) for i in range(n_particles)]
            return np.array(j)
        
        
        optimizer = ps.single.GlobalBestPSO(
            n_particles=100, # amount of particles in swarm. 
            dimensions=self.genes, # dimensions equal amount of layers in nn. 
            options={'c1': 0.9, 'c2': 0.1, 'w':0.1}, #c1 - cognitive parameter, c2 - social parameter, w - inertia parameter 
            bounds=(np.array([-10] * self.genes), np.array([10] * self.genes)) #dimemsions. 
            )

        cost, pos = optimizer.optimize(fit, iters=self.iterations)
        plot_cost_history(cost_history=optimizer.cost_history)
        plt.show()
        self.mlp = MLP(self.structure, weights = pos, activationType=self.activationType)
        return self.mlp  
        



# creates data
# either vectors or XOR classification.
def createData(dataType):
    
    if dataType == "vector":
        data = np.array([[np.random.random()/2 for _ in range(6)] for _ in range(1000)])
        labels = np.array([[i[0] + i[2] + i[4], i[1] + i[3] + i[5] ] for i in data])
        inputs = 6
        outputs = 2
        testData = np.array([[np.random.random()/2 for _ in range(6)] for _ in range(1)])
        testLabels = np.array([[i[0] + i[2] + i[4], i[1] + i[3] + i[5] ] for i in testData])
    
    elif dataType == "XOR":
        data = np.zeros((1000, 8))
        labels = np.zeros((1000, 4))
        tempData = np.array([[0,0],[0,1],[1,0],[1,1]])
        tempLabels = np.array([0,1,1,0])
        for i in range(1000):
            assert len(tempData) == len(tempLabels)
            shuffled = np.random.permutation(len(tempData))
            print(tempData[shuffled].flatten() )
            data[i] = tempData[shuffled].flatten() 
            labels[i] = tempLabels[shuffled] 
        print(data)
        inputs = 8
        outputs = 4
        testData = np.array([[0,0,0,1,1,0,1,1]])
        testLabels = np.array([0,1,1,0])
    else:
        print("No data type")
    

    return data, labels, testLabels, testData, inputs, outputs
    

def main(perceptronData, data, activation, perceptronD = None, perceptronL = None):
    
    data, labels, testLabels, testData, inputs, outputs = createData(data)
    structure = [inputs, 32, outputs] # Creates structure for nn

    env = Enviroment(structure, activation, data, labels, 1000, 0.1)
    
    if perceptronData == "load":
        env.load_data()
    else:
        env.install_data(perceptronD, perceptronL)
    env.runPerceptron()

    env.runBackProp() 
    env.test(testData, testLabels)
    env.runGA()
    env.test(testData, testLabels)
    env.runPSO()
    env.test(testData, testLabels)

        



if __name__ == "__main__":
    # Perceptron Data ('load', 'install')
    perceptronData = "load"
    if perceptronData == "load":
        perData = None
        perLabel = None

    # Data for nn ("vector", "XOR")
    data = "vector"

    # Activation function ("sigmoid", "tanh")
    activation = "sigmoid"


    
    #If you want to use your own classification problem you can include it here.
    #Select perceptronData = "install" 
    #Training data should be called data
    #Labels should be called label and be in a binary format. 
    
    
    main(perceptronData, data, activation, perceptronD = perData, perceptronL = perLabel)