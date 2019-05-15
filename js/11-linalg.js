
/** ------------------------------------------------------

Linear Algebra library used by the Linear and Neural
Network classifiers.

--------------------------------------------------------- */

/**
    Vector implementation (1D array).
*/
class NN_vec1D {
    
    // Creates a new vector filled with 0
    static zeros(s) {
        let v = new Array(s).fill(0);
        return new NN_vec1D(v);
    }
    
    // Creates a new vector with random values scaled between -scale ... scale
    static random(s, scale=1) {
        //Generate random double values between: 0 ... 1
        let min = 10000;
        let max = -10000;
        let v = new Array(s).fill(0);
        for (let a = 0; a < s; a++) {
            v[a] = rnd() * scale;
            if (v[a] < min) min = v[a];
            if (v[a] > max) max = v[a];
        }
        
        //Normalize values between: -scale ... scale
        let sv = new Array(s).fill(0);
        for (let a = 0; a < s; a++) {
            sv[a] = (v[a] - min) / (max - min) * scale * 2 - scale;
        }
        
        return new NN_vec1D(sv);
    }
    
    // Constructor
    constructor(v) {
        this.v = v;
    }
    
    // Returns a value in the vector
    get(i) {
        return this.v[i];
    }
    
    // Sets a value in the vector
    set(i, val) {
        this.v[i] = val;
    }
    
    // Returns the size of the vector
    size() {
        return this.v.length;
    }
    
    // Adds a scaled vector to this vector
    add_vector(v1, scale) {
        if (this.size() != v1.size()) {
            throw "Vector sizes must be equal";
        }
        
        for (let a = 0; a < this.size(); a++) {
            this.v[a] += v1.v[a] * scale;
        }
    }
    
    // Returns the sum of all values in the vector
    sum() {
        let sum = 0;
        for (let a = 0; a < this.size(); a++) {
            sum += this.v[a];
        }
        return sum;
    }
    
    // Returns the max value in the vector
    max() {
        let max = -100000;
        for (let a = 0; a < this.size(); a++) {
            if (this.v[a] > max) {
                max = this.v[a];
            }
        }
        return max;
    }
    
    // Returns the min value in the vector
    min() {
        let min = 100000;
        for (let a = 0; a < this.size(); a++) {
            if (this.v[a] < min) {
                min = this.v[a];
            }
        }
        return min;
    }
    
    // Returns the average of the values in the vector
    avg() {
        let avg = 0;
        for (let a = 0; a < this.size(); a++) {
            avg += this.v[a];
        }
        avg /= this.size();
        return avg;
    }
    
    // Calculate the dot-product between this vector and another vector
    dot(v1) {
        if (this.size() != v1.size()) {
            throw "Vector sizes must be equal";
        }
        
        let dot = 0;
        for (let a = 0; a < this.size(); a++) {
            dot += this.v[a] * v1.v[a];
        }
        return dot;
    }
    
    // Multiplies a row vector with a column vector
    static mul(v1, v2) {
        if (v1.size() != v2.size()) {
            throw "Vector sizes must be equal";
        }
        
        let nv = NN_vec2D.create_vec2D(v1.size(), v1.size());
        for (let r = 0; r < v1.size(); r++) {
            for (let c = 0; c < v2.size(); c++) {
                nv[r][c] = v1.v[r] * v2.v[c];
            }
        }
        
        return nv; //TODO: New NN_vec2D
    }
    
    // Returns the index of the highest value in the vector
    argmax() {
        let mV = this.v[0];
        let mI = 0;
        
        for (let i = 0; i < this.size(); i++) {
            if (this.v[i] > mV) {
                mV = this.v[i];
                mI = i;
            }
        }
        
        return mI;
    }
    
    // Updates the weights, assuming this is a bias vector
    update_weights(dB, learningrate) {
        for (let i = 0; i < this.size(); i++) {
            this.v[i] -= dB.get(i) * learningrate;
        }
    }
    
    // Creates a copy of this vector
    copy() {
        let v1 = NN_vec1D.zeros(this.size());
        for (let i = 0; i < this.size(); i++) {
            v1.set(i, this.v[i]);
        }
        return v1;
    }
    
    // Prints this vector in the console
    print() {
        let str = "";
        for (let a = 0; a < this.size(); a++) {
            str += this.v[a].toFixed(2) + " ";
        }
        console.log(str);
    }
}

/**
    Matrix implementation (2D array).
*/
class NN_vec2D {
    
    /**
        Creates an empty matrix.
    */
    static create_vec2D(r, c) {
        let v = new Array(r);
        for (let i = 0; i < r; i++) {
            v[i] = new Array(c).fill(0);
        }
        return v;
    }

    // Creates a new matrix filled with 0
    static zeros(r, c) {
        let v = NN_vec2D.create_vec2D(r, c);
        return new NN_vec2D(v);
    }
    
    // Creates a new matrix with random values scaled between -scale ... scale
    static random(r, c, scale=1) {
        //Generate random double values between: 0 ... 1
        let min = 1000;
        let max = -1000;
        let v = NN_vec2D.create_vec2D(r, c);
        for (let a = 0; a < r; a++) {
            for (let b = 0; b < c; b++) {
                v[a][b] = rnd();
                if (v[a][b] < min) min = v[a][b];
                if (v[a][b] > max) max = v[a][b];
            }
        }
        
        //Normalize values between: -scale ... scale
        let sv = NN_vec2D.create_vec2D(r, c);
        for (let a = 0; a < r; a++) {
            for (let b = 0; b < c; b++) {
                sv[a][b] = (v[a][b] - min) / (max - min) * scale * 2 - scale;
            }
        }
        
        return new NN_vec2D(sv);
    }
    
    // Creates a new matrix with random normally distributed values
    static randomNormal(r, c) {
        //Desired standard deviation:
        //2.0/sqrt(noInputs)
        let stddev = 2 / Math.sqrt(c);
        
        //Generate random double values between: 0 ... 1
        let v = NN_vec2D.create_vec2D(r, c);
        for (let a = 0; a < r; a++)
        {
            for (let b = 0; b < c; b++)
            {
                v[a][b] = rnd_bm() * stddev;
            }
        }
        
        return new NN_vec2D(v);
    }
    
    // Constructor
    constructor(v) {
        this.v = v;
    }
    
    // Returns the number of rows in this matrix
    rows() {
        return this.v.length;
    }
    
    // Returns the number of columns in this matrix
    columns() {
        return this.v[0].length;
    }
    
    // Returns a value in the matrix
    get(r, c) {
        return this.v[r][c];
    }
    
    // Sets a value in the matrix
    set(r, c, val) {
        this.v[r][c] = val;
    }
    
    // Returns a column in the matrix
    get_column(c) {
        let nv = new Array(this.columns()).fill(0);
        for (let r = 0; r < this.rows(); r++) {
            nv[r] = this.v[r][c];
        }
    }
    
    // Adds a value to this matrix
    add(r, c, val) {
        this.v[r][c] += val;
    }
    
    // Calculates activation (w*x+b) for a weights matrix and an input vector
    static activation_1D(w, x, b) {
        //Error checks
        if (w.columns() != x.size()) {
            console.log("A");
            throw "Number of columns in weights matrix does not match size of input vector";
        }
        if (w.rows() != b.size()) {
            console.log("B");
            throw "Number of rows in weights matrix does not match size of bias vector";
        }
        
        //Activation vector
        let nv = new Array(w.rows()).fill(0);
        
        for (let r = 0; r < w.rows(); r++) {
            //Multiply the row in weight matrix with the input vector
            for (let c = 0; c < w.columns(); c++) {
                nv[r] += w.get(r, c) * x.get(c);
            }
            //Add bias
            nv[r] += b.get(r);
        }
        
        return new NN_vec1D(nv);
    }
    
    // Calculates activation (w*x+b) for a weights matrix and an input matrix
    static activation(w, x, b) {
        //Error checks
        if (w.columns() != x.rows()) {
            throw "Number of columns in weights matrix does not match rows of input matrix";
        }
        if (w.rows() != b.size())
        {
            throw "Number of rows in weights matrix does not match size of bias matrix";
        }
        
        //Activation matrix
        let nv = NN_vec2D.create_vec2D(w.rows(), x.columns());
        
        for (let nc = 0; nc < x.columns(); nc++) {
            for (let r = 0; r < w.rows(); r++) {
                //Multiply the row in weight matrix with the input matrix
                for (let c = 0; c < w.columns(); c++) {
                    nv[r][nc] += w.v[r][c] * x.v[c][nc];
                }
                //Add bias
                nv[r][nc] += b.v[r];
            }
        }
        
        return new NN_vec2D(nv);
    }
    
    // Calculates the product of the transpose of matrices w and d
    static transpose_mul(w, d) {
        //Error checks
        if (w.rows() != d.rows()) {
            throw "Number of rows in matrix w does not match rows of matrix d";
        }
        
        //Result matrix
        let nv = NN_vec2D.create_vec2D(w.columns(), d.columns());
        
        for (let nc = 0; nc < d.columns(); nc++) {
            for (let r = 0; r < w.columns(); r++) {
                for (let c = 0; c < w.rows(); c++) {
                    nv[r][nc] += w.v[c][r] * d.v[c][nc]; //Exchange rows with cols in w to get transpose
                }
            }
        }
        
        return new NN_vec2D(nv);
    }
    
    // Multiplies a matrix with the transpose of another matrix
    static mul_transpose(d, x) {
        //Error checks
        if (d.columns() != x.columns()) {
            throw "Number of columns in matrix d does not match columns of matrix x";
        }
        
        //Result matrix
        let nv = NN_vec2D.create_vec2D(d.rows(), x.rows());
        
        for (let nc = 0; nc < x.rows(); nc++) {
            for (let r = 0; r < d.rows(); r++) {
                for (let c = 0; c < d.columns(); c++) {
                    nv[r][nc] += d.v[r][c] * x.v[nc][c]; //Exchange rows with cols in x to get transpose
                }
            }
        }
        
        return new NN_vec2D(nv);
    }
    
    // Multiplies two matrices
    static mul(m1, m2) {
        //Error checks
        if (m1.columns() != m2.rows()) {
            throw "Number of columns in matrix m1 does not match rows of matrix m2";
        }
        
        //Result matrix
        let nv = NN_vec2D.create_vec2D(m1.rows(), m2.columns());
        
        for (let nc = 0; nc < m2.columns(); nc++) {
            for (let r = 0; r < m1.rows(); r++) {
                //Multiply the row in m1 with the columns in m2
                for (let c = 0; c < m1.columns(); c++) {
                    nv[r][nc] += m1.v[r][c] * m2.v[c][nc];
                }
            }
        }
        
        return new NN_vec2D(nv);
    }
    
    // Calculates the L2 norm (sum of all squared values) for this matrix
    L2_norm() {
        let norm = 0;
        
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++) {
                norm += Math.pow(this.v[r][c], 2);
            }
        }
        
        return norm;
    }
    
    // For each column, subtracts the values by the max value of that column resulting in all values being less than or equal to zero
    shift_columns() {
        for (let c = 0; c < this.columns(); c++) {
            let max = -100000;
            for (let r = 0; r < this.rows(); r++) {
                if (this.v[r][c] > max) max = this.v[r][c];
            }
            
            //Shift values
            for (let r = 0; r < this.rows(); r++) {
                this.v[r][c] -= max;
            }
        }
        
        return this;
    }
    
    // Normalizes each column in the matrix so the sum of each column is 1
    normalize() {
        for (let c = 0; c < this.columns(); c++) {
            //Calculate sum
            let sum = 0;
            for (let r = 0; r < this.rows(); r++) {
                sum += this.v[r][c];
            }
            
            //Normalize values
            for (let r = 0; r < this.rows(); r++) {
                this.v[r][c] /= sum;
            }
        }
    }
    
    // Calculates the dscores matrix from normalized log probabilities vector
    calc_dscores(y) {
        for (let c = 0; c < this.columns(); c++) {
            // Find correct label for this training example
            let corr_index = y.get(c);
            // Subtract the column value by 1
            this.v[corr_index][c] -= 1.0;
            
            // Divide by number of training examples
            for (let r = 0; r < this.rows(); r++) {
                this.v[r][c] /= y.v.length;
            }
        }
        
        return this;
    }
    
    // Calculates the Softmax cross-entropy loss vector for this matrix
    calc_loss(y) {
        //Loss values
        let L = new Array(y.length).fill(0);
        
        for (let c = 0; c < this.columns(); c++) {
            // Find correct class score for this training example
            let class_score = this.v[y.get(c)][c];
            // Calculate loss
            let Li = -1.0 * Math.log(class_score) / Math.log(Math.E);
            L[c] = Li;
        }
        
        return new NN_vec1D(L);
    }
    
    // Calculates E^v for all values in this matrix
    exp() {
        for (let c = 0; c < this.columns(); c++) {
            for (let r = 0; r < this.rows(); r++) {
                this.v[r][c] = Math.pow(Math.E, this.v[r][c]);
            }
        }
        
        return this;
    }
    
    // Creates a new vector with the sum of each row in this matrix
    sum_rows() {
        let sum = new Array(this.rows()).fill(0);
        
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++) {
                sum[r] += this.v[r][c];
            }
        }
        
        return new NN_vec1D(sum);
    }
    
    // Adds a scaled matrix to this matrix
    add_matrix(m, scale) {
        //Error checks
        if (this.rows() != m.rows() || this.columns() != m.columns()) {
            throw "Size of matrices does not match";
        }
        
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++)
            {
                this.v[r][c] += m.get(r,c) * scale;
            }
        }
    }
    
    // Updates the weights, assuming this is a weights matrix
    update_weights(dW, learningrate) {
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++) {
                this.v[r][c] -= dW.get(r,c) * learningrate;
            }
        }
    }
    
    // Calculates the index for the highest value in a column
    argmax(c) {
        let high = -100000;
        let index = -1;
        
        for (let r = 0; r < this.rows(); r++) {
            if (this.v[r][c] > high) {
                high = this.v[r][c];
                index = r;
            }
        }
        
        return index;
    }
    
    // Divides all values in this matrix by a constant
    div(cons) {
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++) {
                this.v[r][c] /= cons;
            }
        }
    }
    
    // Performs the max operation on all values in the matrix
    max(max_val) {
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++) {
                this.v[r][c] = Math.max(this.v[r][c], max_val);
            }
        }
    }
    
    // Backpropagates the ReLU non-linearity into the gradients matrix
    // All gradient values are set to 0 if the corresponding activation value is 0
    backprop_relu(scores) {
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++) {
                //Check if activation is <= 0
                if (scores.get(r, c) <= 0) {
                    //Switch off
                    this.v[r][c] = 0;
                }
            }
        }
    }
    
    // Creates a copy of this matrix
    copy() {
        let v1 = NN_vec2D.zeros(this.rows(), this.columns());
        for (let r = 0; r < this.rows(); r++) {
            for (let c = 0; c < this.columns(); c++) {
                v1.set(r, c, this.get(r, c));
            }
        }
        return v1;
    }
    
    // Prints this matrix in the console
    print() {
        let str = "";
        for (let a = 0; a < this.rows(); a++) {
            for (let b = 0; b < this.columns(); b++) {
                str += this.v[a][b].toFixed(2) + " ";
            }
            str += "\n";
        }
        console.log(str);
    }
}

/**
    Creates and returns the transpose of a matrix.
*/
function transpose(x) {
    let nx = NN_vec2D.zeros(x[0].length, x.length);
    for (let r = 0; r < x.length; r++) {
        for (let c = 0; c < x[0].length; c++) {
            nx.set(c, r, x[r][c]);
        }
    }
    return nx;
}
