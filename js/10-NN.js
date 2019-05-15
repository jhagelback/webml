
/** ------------------------------------------------------

Implementation of Neural Network classifier.

See explanation here:
http://cs231n.github.io/optimization-1/
http://cs231n.github.io/optimization-2/
http://cs231n.github.io/neural-networks-1/

--------------------------------------------------------- */

class NeuralNetwork {
    // Constructor
    constructor(layers, iterations, learningrate, batch_size, stop_threshold=0.0, L2_regularization=0.0, momentum=0.1, dropout=0.2) {
        // Number of unique labels
        this.noCategories = 0;
        // Number of input values
        this.noInputs = 0;
        // Input data
        this.x = null;
        // Data labels
        this.y = null;
        // Hidden layers
        this.hidden = null;
        // Output layer
        this.out = null;
        // Current training iteration
        this.current_iter = 0;
        // Training done
        this.training_done = false;
        // Current loss
        this.loss = 0;
        // Previous loss
        this.prev_loss = 0;
        // Batch size for batch training
        this.batch_size =batch_size;
        // Settings: hidden layers
        this.layers = layers;
        // Settings: Number of training iterations
        this.iterations = iterations;
        // Settings: Loss threshold for stopping training
        this.stop_threshold = stop_threshold;
        // Settings: Learning rate
        this.learningrate = learningrate;
        // Settings: L2 regularization
        this.L2_regularization = L2_regularization;
        // Settings: Momenum
        this.momentum = momentum;
        // Settings: Dropout
        this.dropout = dropout;
        // Init seed
        seed = 42;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return true;
    }
    
    // Initalizes the weights and biases
    init() {
        // Hidden layers
        this.hidden = new Array(this.layers.length);
        this.hidden[0] = new HiddenLayer(this.noInputs, this.layers[0], this.learningrate, this.L2_regularization, this.momentum, this.dropout);
        if (this.hidden.length > 1) {
            for (let i = 1; i < this.hidden.length; i++) {
                this.hidden[i] = new HiddenLayer(this.layers[i-1], this.layers[i], this.learningrate, this.L2_regularization, this.momentum, this.dropout);
            }
        }
        // Out layer
        this.out = new OutLayer(this.layers[this.layers.length - 1], this.noCategories, this.learningrate, this.L2_regularization, this.momentum);
    }
    
    // Trains the classifier with attributes (x) and known labels (y)
    train(data) {
        this.x = transpose(data.x);
        this.y = new NN_vec1D(data.y);
        
        this.noCategories = data.no_classes();
        this.noInputs = data.no_attr();
        this.training_done = false;
        this.current_iter = 0;
        
        //Initializes weights and biases
        this.init();
    }
    
    // Checks if the training is done or not
    done() {
        return this.training_done;
    }
    
    // Returns the current iteration
    current_iteration() {
        return this.current_iter;
    }
    
    // Executes several training iterations
    iterate_steps(stepsize) {
        for (let i = 0; i < stepsize; i++) {
            this.iterate();
        }
    }

    // Executes all training iterations
    iterate_all() {
        while(!this.training_done) {
            this.iterate();
        }
    }
    
    // Executes one training iteration
    iterate() {
        // Sets previous loss
        this.prev_loss = this.loss;
        
        // Batch training
        let bx = this.x;
        let by = this.y;
        if (this.batch_size > 0) {
            let batch = get_batch(this.x, this.y, this.batch_size);
            bx = batch[0];
            by = batch[1];
        }

        //Forward pass (activation)
        this.forward(bx);

        // Dropout
        if (this.dropout > 0.0) {
            for (let i = 0; i < this.hidden.length; i++) {
                this.hidden[i].apply_dropout();
            }
        }
        
        // Backward pass
        this.backward(by);

        // Calculate loss
        this.forward(this.x);

        // We only need to take loss from output layer into consideration, since
        // loss on hidden layers are purely based on regularization
        this.loss = this.out.backward(this.y);

        // Current training iteration
        this.current_iter++;
        if (this.current_iter >= this.iterations) {
            this.training_done = true;
        }
        
        //Check stopping criterion
        if (this.current_iter > 2) {
            let diff = Math.abs(this.loss - this.prev_loss);
            if (diff <= this.stop_threshold && diff >= 0) {
                this.training_done = true;
            }
        }
    }
    
    // Predicts a list of instances
    predict(instances) {
        // Perform forward pass
        this.forward_new(instances);
        let pred = [];
        for (let i = 0; i < instances.length; i++) {
            pred.push(this.classify(i));
        }
        return pred;
    }
    
    // Classifies (predicts) an instance
    classify(i) {
        let pred_class = this.out.scores.argmax(i);
        return pred_class;
    }
    
    // Performs the forward pass (activation)
    forward(x1) {
        // First hidden layer (data as input)
        this.hidden[0].forward(x1);
        // Subsequent hidden layers (previous hidden layer as input)
        for (let i = 1; i < this.hidden.length; i++) {
            this.hidden[i].forward(this.hidden[i-1].scores);
        }
        // Output layer (last hidden layer as input)
        this.out.forward(this.hidden[this.hidden.length-1].scores);
    }
    
    // Performs the forward pass for new data
    forward_new(x1) {
        // Input data
        // Must be transposed
        let nX = NN_vec2D.zeros(x1[0].length, x1.length);
        for (let r = 0; r < x1.length; r++) {
            for (let c = 0; c < x1[0].length; c++) {
                nX.set(c, r, x1[r][c]);
            }
        }
        
        // First hidden layer (data as input)
        this.hidden[0].forward(nX);
        // Subsequent hidden layers (previous hidden layer as input)
        for (let i = 1; i < this.hidden.length; i++) {
            this.hidden[i].forward(this.hidden[i-1].scores);
        }
        // Output layer (last hidden layer as input)
        this.out.forward(this.hidden[this.hidden.length-1].scores);
    }
    
    // Performs the backwards pass
    backward(y1) {
        // Output layer
        let loss = this.out.backward(y1);
        // Last hidden layer (gradients from output layer)
        loss += this.hidden[this.hidden.length-1].backward(this.out.w, this.out.dscores);
        // Rest of the hidden layers (gradients from next layer)
        for (let i = this.hidden.length - 2; i >= 0; i--) {
            loss += this.hidden[i].backward(this.hidden[i+1].w, this.hidden[i+1].dhidden);
        }
        
        // Weights updates
        this.out.updateWeights();
        for (let i = this.hidden.length - 1; i >= 0; i--) {
            this.hidden[i].updateWeights();
        }
        
        return loss;
    }
}

/**
    Hidden Layer.
*/
class HiddenLayer {
    // Constructor
    constructor(noInputs, noOutputs, learningrate, L2_regularization, momentum, dropout) {
        // Number of outputs
        this.noOutputs = noOutputs;
        // Number of input values
        this.noInputs = noInputs;
        // Weights matrix
        this.w = null;
        // Bias vector
        this.b = null;
        // Input data
        this.x = null;
        // Gradients for gradient descent optimization
        this.dW = null;
        this.dB = null;
        // Scores matrix = X*W+b
        this.scores = null;
        // ReLU gradients tensor
        this.dhidden = null;
        // L2 regularization
        this.RW = 0;
        
        // Settings: Learning rate
        this.learningrate = learningrate;
        // Settings: L2 regularization strength
        this.lambda = L2_regularization;
        // Settings: Sets momentum rate (0.0 for no momentum).
        this.momentum = momentum;
        // Settings: dropout
        this.dropout = dropout;

        // Initializes the layer
        this.init();
    }
    
    // Initalizes the weights and biases
    init() {
        // Initialize weights matrix and bias vector
        this.w = NN_vec2D.randomNormal(this.noOutputs, this.noInputs);
        this.b = NN_vec1D.zeros(this.noOutputs);
    }
    
    // Creates a copy of this layer
    copy() {
        let nh = new HiddenLayer(this.noInputs, this.noOutputs);
        nh.w = this.w.copy();
        nh.b = this.b.copy();
        return nh;
    }
    
    // Performs the forward pass (activation)
    forward(x) {
        this.x = x;
        
        // Activation
        this.scores = NN_vec2D.activation(this.w, this.x, this.b);
        // ReLU activation
        this.scores.max(0);
    }
    
    // Applies dropout to this layer.
    // Dropout is a regularization where we zero-out random units during the training phase.
    apply_dropout() {
        for (let r = 0; r < this.scores.rows(); r++) {
            if (rnd() < this.dropout) {
                for (let c = 0; c < this.scores.columns(); c++) {
                    this.scores.set(r, c, 0);
                }
            }
        }
    }
    
    // Performs the backwards pass
    backward(w2, dscores) {
        // Evaluate gradients
        this.grad_relu(w2, dscores);
        
        return 0.5 * this.RW;
    }
    
    // Calculates gradients
    grad_relu(w2, dscores) {
        // Re-calculate regularization
        this.calc_regularization();
        
        // Backprop into hidden layer
        this.dhidden = NN_vec2D.transpose_mul(w2, dscores);
        //Backprop the ReLU non-linearity (set dhidden to 0 if activation is 0
        this.dhidden.backprop_relu(this.scores);
        
        // Momentum
        let oldDW = null;
        let oldDB = null;
        if (this.dW != null && this.momentum > 0.0) {
            oldDW = this.dW.copy();
            oldDB = this.dB.copy();
        }
        
        // ... and finally the gradients
        this.dW = NN_vec2D.mul_transpose(this.dhidden, this.x);
        this.dB = this.dhidden.sum_rows();
        
        // Momentum
        if (oldDW != null && this.momentum > 0.0) {
            this.dW.add_matrix(oldDW, this.momentum);
            this.dB.add_vector(oldDB, this.momentum);
        }
        
        // Add regularization to gradients
        // The weight tensor scaled by Lambda*0.5 is added
        if (this.lambda > 0) {
            this.dW.add_matrix(this.w, this.lambda * 0.5);
        }
    }
    
    // Updates the weights and bias tensors.
    updateWeights() {
        // Update weights
        this.w.update_weights(this.dW, this.learningrate);
        // Update bias
        this.b.update_weights(this.dB, this.learningrate);
    }
    
    // Re-calcualtes the L2 regularization loss
    calc_regularization() {
        //Regularization
        this.RW = 0;
        
        if (this.lambda > 0) {
            this.RW = this.w.L2_norm() * this.lambda;
        }
    }
}

/**
    Output Layer.
*/
class OutLayer {
    // Constructor
    constructor(noInputs, noCategories, learningrate, L2_regularization, momentum) {
        // Number of unique labels
        this.noCategories = noCategories;
        // Number of input values
        this.noInputs = noInputs;
        // Weights matrix
        this.w = null;
        // Bias vector
        this.b = null;
        // Input data
        this.x = null;
        // Data labels
        this.y = null;
        // Gradients for gradient descent optimization
        this.dW = null;
        this.dB = null;
        // Scores matrix = X*W+b
        this.scores = null;
        // ReLU gradients matrix
        this.dhidden = null;
        // L2 regularization
        this.RW = 0;
        
        // Settings: Learning rate
        this.learningrate = learningrate;
        // Settings: L2 regularization strength
        this.lambda = L2_regularization;
        // Settings: Sets momentum rate (0.0 for no momentum).
        this.momentum = momentum;

        // Initializes the layer
        this.init();
    }
    
    // Initalizes the weights and biases
    init() {
        // Initialize weights matrix and bias vector
        this.w = NN_vec2D.randomNormal(this.noCategories, this.noInputs);
        this.b = NN_vec1D.zeros(this.noCategories);
    }
    
    // Creates a copy of this layer
    copy() {
        let nh = new OutLayer(this.noInputs, this.noCategories);
        nh.w = this.w.copy();
        nh.b = this.b.copy();
        return nh;
    }
    
    // Performs the forward pass (activation)
    forward(x) {
        this.x = x;
        
        // Activation
        this.scores = NN_vec2D.activation(this.w, this.x, this.b);
    }
    
    // Performs the backwards pass
    backward(y) {
        // Input data
        this.y = y;
        
        // Calculate loss and evaluate gradients
        let loss = this.grad_softmax();
        
        return loss;
    }
    
    // Classifies an instance
    classify(i) {
        let pred_class = this.scores.argmax(i);
        return pred_class; 
    }
    
    // Evaluates loss and calculates gradients using Softmax
    grad_softmax() {
        // Re-calculate regularization
        this.calc_regularization();
        
        // Init some variables
        let num_train = this.x.columns();
        let loss = 0;
        
        // Calculate exponentials
        // To avoid numerical instability
        let logprobs = this.scores.shift_columns();
        logprobs.exp();
        
        // Normalize
        logprobs.normalize();
        
        // Calculate cross-entropy loss tensor
        let loss_vec = logprobs.calc_loss(this.y);
        
        // Average loss
        loss = loss_vec.sum() / num_train;
        // Regularization loss
        loss += this.RW;
        
        // Momentum
        let oldDW = null;
        let oldDB = null;
        if (this.dW != null && this.momentum > 0.0) {
            oldDW = this.dW.copy();
            oldDB = this.dB.copy();
        }
        
        // Gradients
        this.dscores = logprobs.calc_dscores(this.y);
        this.dW = NN_vec2D.mul_transpose(this.dscores, this.x);
        this.dB = this.dscores.sum_rows();
        
        // Momentum
        if (oldDW != null && this.momentum > 0.0) {
            this.dW.add_matrix(oldDW, this.momentum);
            this.dB.add_vector(oldDB, this.momentum);
        }
        
        // Add regularization to gradients
        // The weight tensor scaled by Lambda*0.5 is added
        if (this.lambda > 0) {
            this.dW.add_matrix(this.w, this.lambda * 0.5);
        }
        
        return loss;
    }
    
    // Calculates data loss using Softmax.
    calc_loss()
    {
        // Init some variables
        let num_train = this.x.columns();
        let loss = 0;
        
        // Calculate exponentials
        // To avoid numerical instability
        let logprobs = this.scores.shift_columns();
        logprobs.exp();
        
        // Normalize
        logprobs.normalize();
        
        // Calculate cross-entropy loss tensor
        let loss_vec = logprobs.calc_loss(this.y);
        
        // Average loss
        loss = loss_vec.sum() / num_train;
        
        return loss;
    }
    
    // Updates the weights and bias
    updateWeights() {
        // Update weights
        this.w.update_weights(this.dW, this.learningrate);
        // Update bias
        this.b.update_weights(this.dB, this.learningrate);
    }
    
    // Re-calcualtes the L2 regularization loss
    calc_regularization() {
        // Regularization
        this.RW = 0;
        
        if (this.lambda > 0) {
            this.RW = this.w.L2_norm() * this.lambda;
        }
    }
}

