
/** ------------------------------------------------------

Implementation of Random Forest classifier.

See explanation here:
https://machinelearningmastery.com/implement-random-forest-scratch-python/

--------------------------------------------------------- */

class RandomForest {
    constructor(estimators, sample_size, max_depth, min_size) {
        this.max_depth = max_depth;
        this.min_size = min_size;
        this.estimators = estimators;
        this.sample_size = sample_size;
        this.forest = [];
        this.labels = null;
        // Init seed
        seed = 42;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return false;
    }
    
    // Trains the model
    train(data) {
        this.labels = data.labels;
        this.init_forest(data);
    }
    
    // Trains the individual trees in the forest
    init_forest(data) {
        for (let i = 0; i < this.estimators; i++) {
            // Use a random subset of the data for each individual
            let sub = this.get_random_subset(data);
            let c = new CART(this.max_depth, this.min_size);
            c.rf = true;
            c.train(sub);
            this.forest.push(c);
        }
    }
    
    // Generates a random subset of the data
    get_random_subset(data) {
        let sub_x = [];
        let sub_y = [];
        
        // No examples
        let size = parseInt(data.no_examples() * this.sample_size);
        
        // Copy examples (examples can appear more than once)
        for (let i = 0; i < size; i++) {
            let index = Math.floor(rnd() * data.no_examples());
            sub_x.push(data.x[index]);
            sub_y.push(data.y[index]);
        }
        return data.create_subset(sub_x, sub_y);
    }
    
    // Predicts a list of instances
    predict(instances) {
        let pred = [];
        for (let i = 0; i < instances.length; i++) {
            let inst = instances[i];
            pred.push(this.classify(inst));
        }
        return pred;
    }
    
    // Classifies (predicts) an instance
    classify(inst) {
        // Run predictors
        this.labels.reset_cnt();
        for (let i = 0; i < this.estimators; i++) {
            let c = this.forest[i];
            this.labels.inc_cnt( c.classify(inst) );
        }
        
        // Return best label
        return this.labels.get_best();
    }
}
