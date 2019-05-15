
/** ------------------------------------------------------

Implementation of CART Decision Tree classifier.

See explanation here:
https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

--------------------------------------------------------- */

class CART {
    constructor(max_depth, min_size) {
        this.max_depth = max_depth;
        this.min_size = min_size;
        this.labels = null;
        this.data = [];
        this.root = null;
        this.rf = false;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return false;
    }
    
    // Trains the model
    train(data) {
        this.labels = data.labels;

        // Create internal dataset
        for (let i = 0; i < data.no_examples(); i++) {
            this.data.push(new Instance(data.x[i], data.y[i]));
        }
        
        this.build_tree(this.max_depth, this.min_size, this.data);
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
        if (this.root == null) {
            return 0;
        }
        
        let node = this.classify_tree(this.root, inst);
        return node.label;
    }
    
    // Recursive classification.
    classify_tree(node, inst) {
        let a_index = node.a_index;
        let val = node.val;
        
        // Check if left or right branch
        if (inst[a_index] < val) {
            if (!node.left.is_terminal()) {
                // Not terminal node - keep iterating
                return this.classify_tree(node.left, inst);
            }
            else {
                // Terminal node - return label
                return node.left;
            }
        }
        else {
            if (!node.right.is_terminal()) {
                // Not terminal node - keep iterating
                return this.classify_tree(node.right, inst);
            }
            else {
                //Terminal node - return label
                return node.right;
            }
        }
    } 
    
    // Recursively builds the CART tree.
    build_tree(m_max_depth, m_min_size, m_data) {
        this.root = this.get_split(m_data);
        this.split(this.root, m_max_depth, m_min_size, 1);
    }
    
    // Tests to split a dataset at the specified attribute and value.
    test_split(m_a_index, m_val, m_data) {
        let left = [];
        let right = [];
        
        for (let i = 0; i < m_data.length; i++) {
            let e = m_data[i];
            if (e.x[m_a_index] < m_val) {
                left.push(e);
            }
            else {
                right.push(e);
            }
        }
        
        let l_node = new CART_node(left, m_a_index, m_val, this.labels);
        let r_node = new CART_node(right, m_a_index, m_val, this.labels);
        return [l_node, r_node];
    }
    
    // Counts the number of instances of the specified class label.
    count(m_data, label) {
        let cnt = 0;
        for (let i = 0; i < m_data.length; i++) {
            let e = m_data[i];
            if (e.label == label) {
                cnt++;
            }
        }
        return cnt;
    }
    
    // Calculates the Gini index for two nodes.
    gini_index(nodes) {
        // Total number of instances
        let n_instances = nodes[0].size() + nodes[1].size();
        // Gini index
        let gini = 0;
        // Iterate over both groups
        for (let i = 0; i < nodes.length; i++) {
            let size = nodes[i].size();
            let score = 0;
            // Calculate score
            if (size > 0) {
                for (let c = 0; c < this.labels.size(); c++) {
                    let e = this.labels.dist[c];
                    let p = this.count(nodes[i].data, e.id) / size;
                    score += Math.pow(p, 2);
                }
                // Update gini index
                gini += (1 - score) * (size / n_instances);
            }
        }
        return gini;
    }
    
    // Randomizes which attributes to evaluate for a split.
    // Used by the random forest classifier to create diverse trees.
    include_attr() {
        let inc = [];
        
        // Must have at least one attribute to consider for each split
        while (inc.length == 0) {
            // 70% chance to include each attribute for random forest
            let p = 1;
            if (this.rf == true) {
                p = 0.7;
            }
            
            let no_attr = this.data[0].x.length;
            for (let a = 0; a < no_attr; a++) {
                let r = rnd();
                if (r <= p) {
                    inc.push(a);
                }
            }
        }
        
        return inc;
    }
    
    // Search for and splits the dataset at the best attribute-value combination.
    get_split(m_data) {
        // Init variables
        let b_index = -1;
        let b_value = 0;
        let b_score = 100000;
        let b_nodes = [];
        let no_attr = m_data[0].x.length;
        
        // Iterate over all attributes...
        let inc = this.include_attr();
        for (let a = 0; a < no_attr; a++) {
            // ... and instances
            if (inc.includes(a)) {
                for (let i = 0; i < m_data.length; i++) {
                    let e = m_data[i];
                    // Current attribute value
                    let m_val = e.x[a];
                    // Test to split at this attribute-value combination
                    let nodes = this.test_split(a, m_val, m_data);
                    // Calculate Gini index for the split
                    let gini = this.gini_index(nodes);
                    // Check if we have a new best split
                    if (gini < b_score) {
                        b_index = a;
                        b_value = m_val;
                        b_score = gini;
                        b_nodes = nodes;
                    }
                }
            }
        }
        
        // Create result node with the dataset splitted into a
        // left and right branch
        let n = new CART_node(m_data, b_index, b_value, this.labels);
        n.left = b_nodes[0];
        n.right = b_nodes[1];
        
        return n;
    }
    
    // Recursive split of the dataset.
    split(node, m_max_depth, m_min_size, depth) {
        // Left and right branch nodes
        let left = node.left;
        let right = node.right;

        // No split since left or right is null
        if (left == null || right == null) {
            //Terminal node - calculate label
            node.calc_label();
            return;
        }
        // Check for max depth
        if (depth >= m_max_depth) {
            // Terminal nodes - calculate labels
            node.left.calc_label();
            node.right.calc_label();
            return;
        }
        // Process left child
        if (left.data.length <= m_min_size) {
            // Terminal node - calculate label
            node.left.calc_label();
        }
        else {
            node.left = this.get_split(left.data);
            // Recursive call
            this.split(node.left, m_max_depth, m_min_size, depth + 1);
        }
        // Process right child
        if (right.data.length <= m_min_size) {
            // Terminal node - calculate label
            node.right.calc_label();
        }
        else {
            node.right = this.get_split(right.data);
            // Recursive call
            this.split(node.right, m_max_depth, m_min_size, depth + 1);
        }
    }
}

/**
    Internal class for tree nodes.
*/
class CART_node {
    constructor(data, a_index, val, labels) {
        // Data subset for this node
        this.data = data;
        // Index of attribute to split at
        this.a_index = a_index;
        // Value to split at
        this.val = val;
        // Left branch
        this.left = null;
        // Right branch
        this.right = null;
        // Label (for leaf nodes)
        this.label = -1;
        // Class distribution (for leaf nodes)
        this.labels = labels;
    }
    
    // Returns the number of instances for this node
    size() {
        return this.data.length;
    }
    
    // Checks if this node is a terminal (leaf) node
    is_terminal() {
        return (this.left == null || this.right == null);
    }
    
    // Calculates the prediced label and class distribution for a terminal node
    calc_label() {
        this.labels.reset_cnt();
        for (let i = 0; i < this.data.length; i++) {
            this.labels.inc_cnt(this.data[i].label);
        }
        this.label = this.labels.get_best();
    }
    
    // Returns the class probabilities
    get_class_probabilities() {
        let p = [];
        // Calculate sum of occurences
        let sum = 0;
        for (let i = 0; i < this.labels.size(); i++) {
            let e = this.labels.dist[i];
            sum += e.no;
        }
        // Calculate probabilities
        for (let i = 0; i < this.labels.size(); i++) {
            let e = this.labels.dist[i];
            let p = e.no / sum;
            p.push( {label: e.label, p: p} );
        }
        return p;
    }
}
