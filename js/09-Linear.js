
/** ------------------------------------------------------

Implementation of Linear Regression Softmax classifier.

See explanation here:
http://cs231n.github.io/linear-classify/

--------------------------------------------------------- */

class LinearRegression {
    // Constructor
    constructor(iterations, learningrate, batch_size, stop_threshold=0.0, L2_regularization=0.0, momentum=0.1) {
        // Number of unique labels
        this.noCategories = 0;
        // Number of input values
        this.noInputs = 0;
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
        // L2 regularization
        this.RW = 0.01;
        // Current training iteration
        this.current_iter = 0;
        // Training done
        this.training_done = false;
        // Current loss
        this.loss = 0;
        // Previous loss
        this.prev_loss = 0;
        // Batch size for batch training
        this.batch_size = batch_size;
        
        // Settings: Number of training iterations
        this.iterations = iterations;
        // Settings: Loss threshold for stopping training
        this.stop_threshold = stop_threshold;
        // Settings: Learning rate
        this.learningrate = learningrate;
        // Settings: L2 regularization strength
        this.lambda = L2_regularization;
        // Settings: Sets momentum rate (0.0 for no momentum).
        this.momentum = momentum;
        // Init seed
        seed = 42;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return true;
    }
    
    // Initalizes the weights and biases
    init() {
        // Initialize weights matrix and bias vector
        this.w = NN_vec2D.randomNormal(this.noCategories, this.noInputs);
        this.b = NN_vec1D.random(this.noCategories);
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
        //this.iterate();
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
        this.activation(bx);
        
        //Calculate loss and evaluate gradients
        this.grad_softmax(bx, by);
        
        //Update weights
        this.updateWeights();
        
        //Calculate loss
        this.activation(this.x);
        this.loss = this.calc_loss(this.x, this.y);
        
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
    
    // Performs forward pass (activation)
    activation(x1) {
        //Activation
        this.scores = NN_vec2D.activation(this.w, x1, this.b);
    }
    
    // Performs forward pass (activation) for new input data
    activation_new(x1) {
        // Input data
        // Must be transposed
        let nX = NN_vec2D.zeros(x1[0].length, x1.length);
        for (let r = 0; r < x1.length; r++) {
            for (let c = 0; c < x1[0].length; c++) {
                nX.set(c, r, x1[r][c]);
            }
        }
        
        //Activation
        this.scores = NN_vec2D.activation(this.w, nX, this.b);
    }
    
    // Predicts a list of instances
    predict(instances) {
        // Perform forward pass
        this.activation_new(instances);
        let pred = [];
        for (let i = 0; i < instances.length; i++) {
            pred.push(this.classify(i));
        }
        return pred;
    }
    
    // Classifies (predicts) an instance
    classify(i) {
        let pred_class = this.scores.argmax(i);
        return pred_class;
    }
    
    // Evaluates loss and calculates gradients using Softmax
    grad_softmax(x1, y1) {
        //Re-calculate regularization
        this.calc_regularization();
        
        //Init some variables
        let num_train = x1.columns();
        let loss = 0;
        
        //Calculate exponentials
        //To avoid numerical instability
        let logprobs = this.scores.shift_columns();
        logprobs.exp();
        
        //Normalize
        logprobs.normalize();
        
        //Calculate cross-entropy loss vector
        let loss_vec = logprobs.calc_loss(y1);
        
        //Average loss
        loss = loss_vec.sum() / num_train;
        //Regularization loss
        loss += this.RW;
        
        //Momentum
        let oldDW = null;
        let oldDB = null;
        if (this.dW != null && this.momentum > 0.0)
        {
            oldDW = this.dW.copy();
            oldDB = this.dB.copy();
        }

        //Gradients
        let dscores = logprobs.calc_dscores(y1);
        this.dW = NN_vec2D.mul_transpose(dscores, x1);
        this.dB = dscores.sum_rows();
        
        //Momentum
        if (oldDW != null && this.momentum > 0.0) {
            this.dW.add_matrix(oldDW, this.momentum);
            this.dB.add_vector(oldDB, this.momentum);
        }
        
        //Add regularization to gradients
        //The weight tensor scaled by Lambda*0.5 is added
        if (this.lambda > 0) {
            this.dW.add_matrix(this.w, this.lambda * 0.5);
        }
        
        return loss;
    }
    
    // Calculates data loss using Softmax
    calc_loss(x1, y1) {
        //Init some variables
        let num_train = x1.columns();
        let loss = 0;
        
        //Calculate exponentials
        //To avoid numerical instability
        let logprobs = this.scores.shift_columns();
        logprobs.exp();
        
        //Normalize
        logprobs.normalize();
        
        //Calculate cross-entropy loss vector
        let loss_vec = logprobs.calc_loss(y1);
        
        //Average loss
        loss = loss_vec.sum() / num_train;
        
        return loss;
    }
    
    // Updates the weights and bias
    updateWeights() {
        //Update weights
        this.w.update_weights(this.dW, this.learningrate);
        //Update bias
        this.b.update_weights(this.dB, this.learningrate);
    }
    
    // Re-calculates the L2 regularization loss
    calc_regularization() {
        //Regularization
        this.RW = 0;
        
        if (this.lambda > 0) {
            this.RW = this.w.L2_norm() * this.lambda;
        }
    }
}
