
/** ------------------------------------------------------

Implementation of simplified Support Vector Machine 
with RBF kernel classifier.

See explanation in the book "Programming
Collective Intelligence" by Toby Segaran, chapter 9.

--------------------------------------------------------- */

class SVM {
    constructor(gamma) {
        this.gamma = gamma;
        this.kernels = [];
        this.labels = null;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return false;
    }
    
    // Trains the model
    train(data) {
        this.labels = data.labels;
        
        // Create kernels for each pair of labels
        for (let i = 0; i < this.labels.size(); i++) {
            for (let j = 0; j < this.labels.size(); j++) {
                if (i < j) {
                    let label0 = this.labels.dist[i].id;
                    let label1 = this.labels.dist[j].id;
                    
                    let k = new RBF_kernel(label0, label1, data.x, data.y, this.gamma);
                    this.kernels.push(k);
                }
            }
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
        // Classify instance with each kernel
        this.labels.reset_cnt();
        for (let i = 0; i < this.kernels.length; i++) {
            let k = this.kernels[i];
            let pred = k.classify(inst);
            this.labels.inc_cnt(pred);
        }
        
        // Return best label
        return this.labels.get_best();
    }
}

/**
    RBF kernel.
*/
class RBF_kernel {
    constructor(label0, label1, x, y, gamma) {
        this.label0 = label0;
        this.label1 = label1;
        this.gamma = gamma;
        this.offset = 0;
        
        // Split the dataset into labels
        this.data0 = [];
        this.data1 = [];
        
        for (let i = 0; i < x.length; i++) {
            if (y[i] == label0) {
                this.data0.push( new Instance(x[i], y[i]) );
            }
            if (y[i] == label1) { //One-vs-one
            //else { //One-vs-all
                this.data1.push( new Instance(x[i], label1) );
            }
        }
        
        this.calc_offset();
    }
    
    // Calculates the offset value
    calc_offset() {
        // Calculate sum of RBF values for label 0
        let s0 = 0;
        for (let i1 = 0; i1 < this.data0.length; i1++) {
            for (let i2 = 0; i2 < this.data0.length; i2++) {
                s0 += this.rbf_value(this.data0[i1].x, this.data0[i2].x);
            }
        }
        
        // Calculate sum of RBF values for label 1
        let s1 = 0;
        for (let i1 = 0; i1 < this.data1.length; i1++) {
            for (let i2 = 0; i2 < this.data1.length; i2++) {
                s1 += this.rbf_value(this.data1[i1].x, this.data1[i2].x);
            }
        }
        
        // Calculate offset
        this.offset = (1 / Math.pow(this.data1.length, 2)) * s1 - (1 / Math.pow(this.data0.length, 2)) * s0;        
    }
    
    // Calculates RBF value for a pair of instances
    rbf_value(x1, x2) {
        let sqDist = 0.0;
        // Find squared distance between x1 and x2
        for (let i = 0; i < x1.length; i++) {
            sqDist += Math.pow(x1[i] - x2[i], 2);
        }
        // Calculate RBF value
        return Math.pow(Math.E, -this.gamma * sqDist);
    }
    
    // Classifies an instance
    classify(inst) {
        //Iterate over all training data instances
        //and calculate RBF values
        // Calculate sum of RBF values for label 0
        let s0 = 0;
        for (let i1 = 0; i1 < this.data0.length; i1++) {
            s0 += this.rbf_value(inst, this.data0[i1].x);
        }
        
        // Calculate sum of RBF values for label 1
        let s1 = 0;
        for (let i1 = 0; i1 < this.data1.length; i1++) {
            s1 += this.rbf_value(inst, this.data1[i1].x);
        }
        
        // Calculate RBF value
        let y = s0 / this.data0.length - s1 / this.data1.length + this.offset;
        
        // Check sign of RBF value to predict category
        if (y > 0) {
            return this.label0;
        }
        else {
            return this.label1;
        }
    }
}
