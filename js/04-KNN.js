
/** ------------------------------------------------------

Implementation of k-Nearest Neighbor classifier.

See explanation here:
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

--------------------------------------------------------- */

class KNN {
    constructor(k, distf) {
        this.k = k;
        this.data = [];
        this.labels = null;
        if (distf == "0") {
            this.dist_func = this.dist_euclidean;
        }
        else if(distf == "1") {
            this.dist_func = this.dist_manhattan;
        }
        else if(distf == "2") {
            this.dist_func = this.dist_chebyshev;
        }
        else {
            throw("Unknown distance function: " + dist);
        }
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return false;
    }
  
    // Trains the classifier with attributes (x) and known labels (y)
    train(data) {
        this.labels = data.labels;
        
        // Add data to internal array
        for (let i = 0; i < data.no_examples(); i++) {
            let e = new Instance(data.x[i], data.y[i]);
            this.data.push(e);
        }
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
        // Calculate distance to each instance in the dataset
        for (let i = 0; i < this.data.length; i++) {
            let e = this.data[i];
            let dist = this.dist_func(inst, e);
            e.dist = dist;
        }
        
        // Sort array
        this.data.sort(sortBy());
        
        // Find label with most occurences among k nearest instances
        this.labels.reset_cnt();
        for (let i = 0; i < this.k; i++) {
            let e = this.data[i];
            this.labels.inc_cnt(e.label);
        }
        
        // Return best label
        return this.labels.get_best();
    }
    
    // Calculates distance (Euclidean) between two instances
    dist_euclidean(inst, e) {
        let sumSq = 0;
        for (let i = 0; i < inst.length; i++) {
            sumSq += Math.pow(inst[i] - e.x[i], 2);
        }
        sumSq = Math.sqrt(sumSq);
        return sumSq;
    }

    // Calculates Manhattan distance between two instances
    dist_manhattan(inst, e) {
        let sum = 0;
        for (let i = 0; i < inst.length; i++) {
            sum += Math.abs(inst[i] - e.x[i]);
        }
        return sum;
    }

    // Calculates Chebyshev distance between two instances
    dist_chebyshev(inst, e) {
        let best_v = 0;
        for (let i = 0; i < inst.length; i++) {
            let v = Math.abs(inst[i] - e.x[i]);
            if (v > best_v) {
                best_v = v;
            }
        }
        return best_v;
    }
}

/**
    Custom sort function.
*/
function sortBy() {
    return function(a,b) {
        return (a.dist > b.dist) - (a.dist < b.dist)
    }
}
