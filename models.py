import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        predicted = nn.as_scalar(self.run(x))
        if predicted >= 0:
            return 1 
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        z = 1 
        while z == 1:
            z = 0
            for x,y in dataset.iterate_once(1):
                predicted = self.get_prediction(x)
                if predicted != nn.as_scalar(y):
                    nn.Parameter.update(self.w,x, nn.as_scalar(y))
                    z = 1

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learn_rate = 0.2
        #20 = layer size
        # input layer
        self.w1 = nn.Parameter(1,20)
        self.b1 = nn.Parameter(1,20)
        # adjacent layer
        self.w2 = nn.Parameter(20,20)
        self.b2 = nn.Parameter(1,20)
        # outputlayer
        self.w3 = nn.Parameter(20,1)
        self.b3 = nn.Parameter(1,1)
        # big enough batch size
        self.batch_size = 200
        # store all layer parameters
        self.w =[self.w1, self.w2, self.w3]
        self.b = [self.b1,self.b2, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # input data is initially x
        input = x
        # loop thorugh based on number of neural network layers
        for i in range(3):
            # nn.Linear(features, weights) ; 
            # use linear regression function to get output node with shape from output features * batch_size
            # from features and weights
            fx = nn.Linear(input,self.w[i])
            output = nn.AddBias(fx, self.b[i])

            # last layer of nn does not need to call (activation) fx
            # activation fx decides whether neuron activates or not
            if (i==2):
                predicted_y=output
            else:
                # if not last layer of nn, call ReLu function, clear out neg entries (unactiv. nodes)
                # calculate input data for next layer
                input = nn.ReLU(output)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # store loss val and loop until loss accuracy reaches required limit
        lossnum = float('inf')
        count = 0
        while lossnum >=0.01:
            # get cobination of (x,y) from dataset as training data
            for (x,y) in dataset.iterate_once(self.batch_size):
                # calculate loss val with loss function
                loss=self.get_loss(x,y)
                lossnum = nn.as_scalar(loss)
                # find gradient
                grads = nn.gradients(loss, self.w +self.b)
                # loop over parameters in gradient to update each weight and bias
                for i in range(3):
                    self.w[i].update(grads[i], -self.learn_rate)
                    self.b[i].update(grads[len(self.w)+i],-self.learn_rate)
                count+=1
            print(count)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learn_rate = 0.4
        
        #20 = layer size
        # input layer
        self.w1 = nn.Parameter(784, 250)
        self.b1 = nn.Parameter(1,250)
        # adjacent layer
        self.w2 = nn.Parameter(250,150)
        self.b2 = nn.Parameter(1,150)
        # adjacent layer
        # self.w3 = nn.Parameter(20,20)
        # self.b3 = nn.Parameter(1,20)
        # outputlayer
        self.w4 = nn.Parameter(150,10)
        self.b4 = nn.Parameter(1,10)
        # big enough batch size
        self.batch_size = 200
        # store all layer parameters
        # self.w =[self.w1, self.w2, self.w3, self.w4]
        # self.b = [self.b1,self.b2, self.b3, self.b4]
        self.w =[self.w1, self.w2, self.w4]
        self.b = [self.b1,self.b2, self.b4]
        # self.w =[self.w1, self.w4]
        # self.b = [self.b1, self.b4]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        input = x
        for i in range(len(self.w)):
            fx = nn.Linear(input,self.w[i])
            output = nn.AddBias(fx, self.b[i])
            if (i==(len(self.w)-1)):
                    predicted_y=output
            else:
                input = nn.ReLU(output)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
         lossnum = float('inf')
        count = 0
        while lossnum >=0.025:
            # get cobination of (x,y) from dataset as training data
            for (x,y) in dataset.iterate_once(self.batch_size):
                # calculate loss val with loss function
                loss=self.get_loss(x,y)
                lossnum = nn.as_scalar(loss)
                # find gradient
                grads = nn.gradients(loss, self.w +self.b)
                # loop over parameters in gradient to update each weight and bias
                for i in range(len(self.w)):
                    self.w[i].update(grads[i], -self.learn_rate)
                    self.b[i].update(grads[len(self.w)+i],-self.learn_rate)
                count+=1
            print(count)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learn_rate = 0.2
        layerSize = 100
        self.batch_size = 200
        # input layer
        self.w1 = nn.Parameter(self.num_chars, layerSize)
        self.b1 = nn.Parameter(1, layerSize)

        # hidden layer
        self.hidden_w = nn.Parameter(layerSize,layerSize)

        # output layer
        self.w2 = nn.Parameter(layerSize,len(self.languages))
        self.b2 = nn.Parameter(1,len(self.languages))

        # store all layer parameters
        self.w = [self.w1,self.w2]
        self.b = [self.b1,self.b2]
        self.parameters = self.w + self.b + [self.hidden_w]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # initial h
        z = nn.Linear(xs[0],self.w1)
        h = nn.ReLU(nn.AddBias(z,self.b1))   
        # subsequent h's, using previous h's
        for i in range(1,len(xs)):
            z = nn.Add(nn.Linear(xs[i],self.w1), nn.Linear(h,self.hidden_w))
            h = nn.ReLU(nn.AddBias(z,self.b1))
        # convert to output size
        return nn.AddBias(nn.Linear(h,self.w2),self.b2)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        logits = self.run(xs)
        return nn.SoftmaxLoss(logits,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        count = 0
        while accuracy < .85:
            for (x,y) in dataset.iterate_once(self.batch_size):
                loss=self.get_loss(x,y)
                grads = nn.gradients(loss, self.parameters)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(grads[i], -self.learn_rate)
                count+=1
                accuracy = dataset.get_validation_accuracy()
