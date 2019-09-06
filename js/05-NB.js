
/** ------------------------------------------------------

Implementation of Gaussian Naive Bayes classifier.

See explanation here:
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python

--------------------------------------------------------- */

class NaiveBayes {
    constructor() {
        this.labels = new NB_labels();
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return false;
    }
  
    // Trains the classifier with attributes (x) and known labels (y)
    train(data) {
        // Update unique labels
        for (let i = 0; i < data.no_examples(); i++) {
            this.labels.update(data.y[i], data.x[i]);
        }

        // Calculate mean and stdev for the attributes separated by class
        this.labels.calc_mean();
        this.labels.calc_stdev();
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
        for (let i = 0; i < this.labels.length(); i++) {
            let e = this.labels.l[i];
            
            let p = 0.0;
            for (let a = 0; a < this.labels.no_attributes(); a++) {
                //p *= this.calc_probability(inst[a], e.mean[a], e.stdev[a]);
                // Use log-probabilities to avoid underflow
                p += Math.log(this.calc_probability(inst[a], e.mean[a], e.stdev[a]));
            }

            e.prob = Math.exp(p);
        }

        // Return best label
        return this.labels.get_best();
    }
    
    // Calculates the probability of an attributed value belonging to a class
    calc_probability(x, mean, stdev) {
        let exponent = Math.exp(-(Math.pow(x - mean, 2) / (2 * Math.pow(stdev, 2))));
        return (1 / (Math.sqrt(2 * Math.PI) * stdev)) * exponent;
    }
}

/**
    Internal class for holding the dataset separated by class.
*/
class NB_labels {
    constructor() {
        this.l = [];
        this.no_attr = 0;
    }
    
    // Updates with a new instances
    update(id, x) {
        for (let i = 0; i < this.l.length; i++) {
            let e = this.l[i];
            if (e.id == id) {
                e.data.push(x);
                return;
            }
        }

        // Add new label entry
        let mean_arr = new Array(x.length).fill(0);
        let stdev_arr = new Array(x.length).fill(0);
        this.l.push( {id: id, data: [x], mean: mean_arr, stdev: stdev_arr, prob: 0} );

        // Set number of attributes
        this.no_attr = x.length;
    }

    // Calculates the mean values for all variables separated by class
    calc_mean() {
        for (let i = 0; i < this.l.length; i++) {
            let e = this.l[i];

            let data = e.data;
            for (let r = 0; r < data.length; r++) {
                for (let c = 0; c < this.no_attr; c++) {
                    e.mean[c] += data[r][c];
                }
            }
            for (let c = 0; c < this.no_attr; c++) {
                e.mean[c] /= data.length;
            }
        }
    }

    // Calculates the standard deviations for all variables separated by class
    calc_stdev() {
        for (let i = 0; i < this.l.length; i++) {
            let e = this.l[i];

            let data = e.data;
            for (let r = 0; r < data.length; r++) {
                for (let c = 0; c < this.no_attr; c++) {
                    e.stdev[c] += Math.pow(data[r][c] - e.mean[c], 2);
                }
            }
            for (let c = 0; c < this.no_attr; c++) {
                e.stdev[c] /= data.length - 1;
                e.stdev[c] = Math.sqrt(e.stdev[c]);
            }
        }
    }

    // Returns the label with highest probability (used for predictions)
    get_best() {
        let max = 0;
        let index = -1;
        
        for (let i = 0; i < this.l.length; i++) {
            if (this.l[i].prob > max || index == -1) {
                max = this.l[i].prob;
                index = i;
            }
        }

        return this.l[index].id;
    }

    // Number of input attributes in this dataset
    no_attributes() {
        return this.no_attr;
    }

    // Number of unique labels
    length() {
        return this.l.length;
    }
}

