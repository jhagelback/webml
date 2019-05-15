
/** ------------------------------------------------------

Code for running ML experiments.

--------------------------------------------------------- */

/**
    File uploader.

    Modified code from:
    https://cmatskas.com/importing-csv-files-using-jquery-and-html5/
*/
$(document).ready(function() {

    // The event listener for the file upload
    let e = document.getElementById('txtFileUpload');
    if (e != null) {
        e.addEventListener('change', upload, false);
    }

    // Method that checks that the browser supports the HTML5 File API
    function browserSupportFileUpload() {
        let isCompatible = false;
        if (window.File && window.FileReader && window.FileList && window.Blob) {
            isCompatible = true;
        }
        return isCompatible;
    }

    // Method that reads and processes the selected file
    function upload(evt) {
        if (!browserSupportFileUpload()) {
        	throw("File upload are not supported in this browser!");
        }
        else {
            let file = evt.target.files[0];
            let reader = new FileReader();
            reader.readAsText(file);
            reader.onload = function(event) {
                let csvData = event.target.result;
                let raw_data = $.csv.toArrays(csvData);
                orig_data = csv_to_data(raw_data);
                if (orig_data != null) {
                    data_name = file.name;
                    document.getElementById("loaded_data").innerHTML = data_name;
                    update_exp_settings();
                }
                else {
                    document.getElementById("loaded_data").innerHTML = "None";
                }
            };
            reader.onerror = function() {
                // Show error
                show("result");
                document.getElementById("result").innerHTML = "<span style='color:red;font-weight:bold;'>ERROR:</span><br>Unable to read " + file.fileName;
            };
        }
    }
});

/**
    Runs an experiment on the uploaded dataset with the specified algorithm and settings.
*/
async function run_experiment() {
    // Wait function
    const delay = ms => new Promise(res => setTimeout(res, ms));
    // Disable button
    disable("demo", "running");
    show("result");
    document.getElementById("result").innerHTML = "<span style='color:blue;'><b>Working...</b></span><br>"
    await delay(25);

    try {
        if (orig_data == null) {
            throw("No dataset uploaded!");
        }

        // Load data        
        let data = orig_data.clone();

        // Data shuffle
        let sh_opt = get_shuffle_opt();
        if (sh_opt == "1") {
            // Set seed for data shuffling
            seed = 42;
            data.shuffle();
        }

        // Data pre-processing
        let pp_opt = get_preprocess_opt();
        if (pp_opt == "1") {
            data.normalize();
        }

        
        // Set seed for classifier
        seed = 42;

        // Get classifier type
        let type = get_selected_classifier();
        
        // Build classifier
        let cl = build_classifier(type);

        // Show dataset metrics
        let str = "<b>File:</b> " + data_name + "<br>";
        str += "<b>Number of examples:</b> " + data.no_examples() + "<br>";
        str += "<b>Number of attributes:</b> " + data.no_attr() + "<br>";
        str += "<b>Number of classes:</b> " + data.no_classes() + "<br><br>";

        // Evaluation option
        let eval_opt = get_eval_opt();
        if (eval_opt == "0") {
            // Train and evaluate on all data
            str = experiment_result(data, data, cl, str);
        }
        else if (eval_opt == "1") {
            // Separate data into training and test sets
            let split_data = data.train_test_split(0.2);
            str = experiment_result(split_data[0], split_data[1], cl, str);
        }
        else if (eval_opt == "2") {
            // 5-fold Cross-Validation
            str = crossval_result(data, cl, str, 5);
        }
        else {
            throw("Invalid experiment option: " + eval_opt);
        }
        
        // Show result
        show("result");
        document.getElementById("result").innerHTML = str;
    }
    catch(err) {
        // Show error
        show("result");
        document.getElementById("result").innerHTML = "<span style='color:red;font-weight:bold;'>ERROR:</span><br>" + err;
    }

    // Enable button
    enable("demo");
}

/**
    Evaluates a classifier.
*/
function experiment_result(training_data, test_data, cl, str) {
    // Train classifier
    let start = new Date().getTime();
    cl.train(training_data);
    if (cl.iterable()) {
        cl.iterate_all();
    }

    let elapsed = (new Date().getTime() - start) / 1000;
    str += "<b>Training time:</b> " + elapsed.toFixed(3) + " sec<br>";

    // Create confusion matrix
    let cm = new ConfusionMatrix(test_data);

    // Make predictions
    start = new Date().getTime();
    let no_correct = 0;
    for (let i = 0; i < test_data.no_examples(); i++) {
        let pred = cl.predict([test_data.x[i]]);
        if (pred == test_data.y[i]) {
            no_correct++;
        }

        cm.add_prediction(test_data.y[i], pred);
    }
    let acc = no_correct / test_data.no_examples() * 100;
    elapsed = (new Date().getTime() - start) / 1000;
    str += "<b>Evaluation time:</b> " + elapsed.toFixed(3) + " sec<br>";
    
    // Show results
    str += "<br><b>Accuracy:</b> " + acc.toFixed(2) + "%&nbsp;&nbsp;(" + no_correct + "/" + test_data.no_examples() + " correctly classified)<br><br>";
    str += cm.get_cm_output();
    str += "<br>";
    str += cm.get_metrics_output();

    return str;
}

/**
    Evaluates a classifier using cross-validation.
*/
function crossval_result(data, cl, str, folds) {
    // Train classifier
    let start = new Date().getTime();

    let cm = new ConfusionMatrix(data);
    let tot_correct = 0;

    str += "<b>Accuracy by fold:</b><br>";

    // Header
    str += "<table><tr>";
    for (let f = 1; f <= folds; f++) {
        str += "<td class='hcm'><center>" + f + "</center></td>";
    }
    str += "</tr><tr>";
    
    // Result for each fold
    for (let f = 1; f <= folds; f++) {
        let cv_data = data.get_fold(f, folds);

        cl.train(cv_data[0]);
        if (cl.iterable()) {
            cl.iterate_all();
        }

        let no_correct = 0;
        for (let i = 0; i < cv_data[1].no_examples(); i++) {
            let pred = cl.predict([cv_data[1].x[i]]);
            if (pred == cv_data[1].y[i]) {
                no_correct++;
            }

            cm.add_prediction(cv_data[1].y[i], pred);
        }
        let acc = no_correct / cv_data[1].no_examples() * 100;
        tot_correct += no_correct;
        
        str += "<td class='cm'>" + acc.toFixed(2) + "%</td>";

        /*str += f + ": " + acc.toFixed(2) + "%";
        if (f < folds) {
            str += ", ";
        } */     
    }

    str += "</tr></table>";

    // Show total accuracy
    let tot_acc = tot_correct / data.no_examples() * 100;
    str += "<br><b>Total accuracy:</b> " + tot_acc.toFixed(2) + "%&nbsp;&nbsp;(" + tot_correct + "/" + data.no_examples() + " correctly classified)<br>";      

    let elapsed = (new Date().getTime() - start) / 1000;
    str += "<br><b>Evaluation time:</b> " + elapsed.toFixed(3) + " sec<br><br>";

    str += cm.get_cm_output();
    str += "<br>";
    str += cm.get_metrics_output();

    return str;
}

/**
    Loads the built-in Iris dataset.
*/
function load_iris() {
    orig_data = get_iris();
    data_name = "iris.csv";
    document.getElementById("loaded_data").innerHTML = data_name;
    update_exp_settings();
}

/**
    Converts a csv file to dataset instance.
*/
function csv_to_data(raw_data) {
    let ds = new Dataset();

    try {
        let no_attr = 0;
        for (let r = 0; r < raw_data.length; r++) {
    		let e = raw_data[r];

    		//X
    		let xe = [];
    		try {
                // Convert x-values
    			for (let c = 0; c < e.length - 1; c++) {
                    xe.push(convert_float(e[c]));
    			}
            }
            catch (e) {
                // Skip row: header or invalid row
            }

            // Label (y-value)
            let ye = e[e.length - 1];

            if (no_attr > 0 && xe.length > 0) {
                if (no_attr != xe.length) {
                    throw("Expected " + no_attr + " attributes, got " + xe.length);
                }
            }

            // Error check
            if (xe.length > 0) {
                ds.add_example(xe, ye);
                no_attr = xe.length;
            }
    	}

        if (ds.no_examples() < 10) {
            throw("Not enough valid examples in csv file (at least 10 is required)");
        }
    }
    catch(err) {
        // Show error
        show("result");
        document.getElementById("result").innerHTML = "<span style='color:red;font-weight:bold;'>ERROR:</span><br>" + err;
        ds = null;
    }

    return ds;
}

/**
    Returns the float value from an input field.
*/
function convert_float(strval) {
    strval = strval.trim();
    let val = parseFloat(strval);
    if (isNaN(val)) {
        throw ("Invalid float: " + strval);
    }
    return val;
}

/**
    Returns the selected evaluation option.
*/
function get_eval_opt() {
    let evalopt = "0";
    let rlist = document.getElementsByName("eval-opt");
    for (let i = 0, length = rlist.length; i < length; i++) {
        if (rlist[i].checked)  {
            evalopt = rlist[i].value;
            break;
        }
    }
    return evalopt;
}

/**
    Returns the selected data pre-processing option.
*/
function get_preprocess_opt() {
    let ppopt = "0";
    let rlist = document.getElementsByName("preproc-data");
    for (let i = 0, length = rlist.length; i < length; i++) {
        if (rlist[i].checked)  {
            ppopt = rlist[i].value;
            break;
        }
    }
    return ppopt;
}

/**
    Returns the selected data shuffle option.
*/
function get_shuffle_opt() {
    let ppopt = "0";
    let rlist = document.getElementsByName("shuffle-data");
    for (let i = 0, length = rlist.length; i < length; i++) {
        if (rlist[i].checked)  {
            ppopt = rlist[i].value;
            break;
        }
    }
    return ppopt;
}

/** 
    Confusion Matrix.
*/
class ConfusionMatrix {

    // Constructor
    constructor(data) {
        this.data = data;
        this.no_classes = data.no_classes();
        
        this.cm = [];
        for (let r = 0; r < this.no_classes; r++) {
            let row = new Array(this.no_classes).fill(0);
            this.cm.push(row);
        }
    }

    // Adds a prediction with known and predicted label
    add_prediction(label, pred) {
        this.cm[label][pred]++;
    }

    // Returns the confusion matrix as HTML code.
    get_cm_output() {
        let str = "<b>Confusion Matrix:</b><br><table>";
        
        // Header
        str += "<tr><td></td>";
        for (let c = 0; c < this.no_classes; c++) {
            str += "<td class='hcm'>[" + c + "]</td>";
        }
        str += "<td></td></tr>";

        // Iterate over all values in the matrix
        for (let r = 0; r < this.no_classes; r++) {
            // Index row
            str += "<tr><td class='hcm'>[" + r + "]</td>";
            // Values
            for (let c = 0; c < this.no_classes; c++) {
                str += "<td class='cm'>" + this.cm[r][c] + "</td>";
            }
            // Label
            str += "<td class='lcm'>&#8594; " + this.data.id_to_label(r) + "</td>";

            str += "</tr>";
        }

        str += "</table>"

        return str;
    }

    // Returns a metrics table as HTML code.
    get_metrics_output() {
        let str = "<b>Metrics by category:</b><br><table>";
        
        // Header
        str += "<tr><td></td><td class='hcm'>Precision</td><td class='hcm'>Recall</td><td class='hcm'>F1 score</td><td></td></tr>";

        // Iterate over all values in the matrix to calculate Precision and Recall
        let avg = [0, 0, 0];
        for (let r = 0; r < this.no_classes; r++) {
            // Index row
            str += "<tr><td class='hcm'>[" + r + "]</td>";
            // Metrics
            let m = this.calc_metrics(r);
            avg[0] += m[0];
            avg[1] += m[1];
            avg[2] += m[2];
            str += "<td class='cm'>" + m[0].toFixed(3) + "</td>";
            str += "<td class='cm'>" + m[1].toFixed(3) + "</td>";
            str += "<td class='cm'>" + m[2].toFixed(3) + "</td>";
            // Label
            str += "<td class='lcm'>&#8594; " + this.data.id_to_label(r) + "</td>";

            str += "</tr>";
        }

        // Average
        avg[0] /= this.no_classes;
        avg[1] /= this.no_classes;
        avg[2] /= this.no_classes;
        str += "<tr><td class='hcm'>Avg:</td>";
        str += "<td class='cm'>" + avg[0].toFixed(3) + "</td>";
        str += "<td class='cm'>" + avg[1].toFixed(3) + "</td>";
        str += "<td class='cm'>" + avg[2].toFixed(3) + "</td>";
        str += "<td></td></tr>";

        str += "</table>";

        return str;
    }

    // Calculates the Precision, Recall and F1 score metrics.
    calc_metrics(r) {
        let tp = this.cm[r][r];
        //let tpfp = this.rowsum(r);
        //let tpfn = this.colsum(r);
        let tpfp = this.colsum(r);
        let tpfn = this.rowsum(r);
        let recall = 0;
        if (tpfn > 0) {
            recall = tp / tpfn;
        }
        let precision = 0;
        if (tpfp > 0) {
            precision = tp / tpfp;
        }
        let f = 0;
        if (precision + recall > 0) {
            f = 2 * precision * recall / (precision + recall);
        }

        return [precision, recall, f];
    }

    // Calculates the sum of a row in the confusion matrix.
    rowsum(r) {
        let s = 0;
        for (let c = 0; c < this.no_classes; c++) {
            s += this.cm[r][c];
        }
        return s;
    }

    // Calculates the sum of a column in the confusion matrix.
    colsum(c) {
        let s = 0;
        for (let r = 0; r < this.no_classes; r++) {
            s += this.cm[r][c];
        }
        return s;
    }
}

/**
    Shows the settings for the selected classifier.
*/
function update_exp_settings() {
    let type = get_selected_classifier();
    
    let settings = get_exp_settings(type);
    
    document.getElementById("opts").innerHTML = "";
    
    let html = "<table><tr>";
    if (type == "knn") {
        html += "<td class='param'>Neighbors:</td><td><input class='value' name='k' id='k' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Distance function:</td><td><select id='dist' class='value'><option value='0'>Euclidean</option><option value='1'>Manhattan</option></option><option value='2'>Chebyshev</option></select></td>";
    }
    else if (type == "nb") {
        html += "<td class='param'>&nbsp;</td>"; 
    }
    else if (type == "linear") {
        html += "<td class='param'>Training iterations:</td><td><input class='value' name='iter' id='iter' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Learning rate:</td><td><input class='value' name='lr' id='lr' value='" + settings[1] + "'></td>";
        html += "<td class='param' title='Batch size for batch training, or 0 for no batch training'>Batch size:</td><td><input class='value' name='batch_size' id='batch_size' value='" + settings[2] + "'></td>";
        html += "</tr><tr>";
        html += "<td class='param'>L2 regularization:</td><td><input class='value' name='L2' id='L2' value='" + settings[3] + "'></td>";
        html += "<td class='param'>Momentum:</td><td><input class='value' name='momentum' id='momentum' value='" + settings[4] + "'></td>";
        html += "<td></td>";
    }
    else if (type == "nn") {
        html += "<td class='param'>Training iterations:</td><td><input class='value' name='iter' id='iter' value='" + settings[1] + "'></td>";
        html += "<td class='param'>Hidden layers:</td><td><input class='value' name='hidden' id='hidden' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Learning rate:</td><td><input class='value' name='lr' id='lr' value='" + settings[2] + "'></td>";
        html += "<td class='param' >Batch size:</td><td><input class='value' name='batch_size' id='batch_size' value='" + settings[3] + "'></td>";
        html += "</tr><tr>";
        html += "<td class='param'>L2 regularization:</td><td><input class='value' name='L2' id='L2' value='" + settings[4] + "'></td>";
        html += "<td class='param'>Momentum:</td><td><input class='value' name='momentum' id='momentum' value='" + settings[5] + "'></td>";
        html += "<td class='param'>Dropout:</td><td><input class='value' name='dropout' id='dropout' value='" + settings[6] + "'></td>";
        html += "<td></td>";
    }
    else if (type == "dt") {
        html += "<td class='param'>Max depth:</td><td><input class='value' name='maxd' id='maxd' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Min samples split:</td><td><input class='value' name='mins' id='mins' value='" + settings[1] + "'></td>"; 
    }
    else if (type == "rf") {
        html += "<td class='param'>No estimators:</td><td><input class='value' name='noe' id='noe' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Max depth:</td><td><input class='value' name='maxd' id='maxd' value='" + settings[1] + "'></td>";
        html += "<td class='param'>Min samples split:</td><td><input class='value' name='mins' id='mins' value='" + settings[2] + "'></td>"; 
        html += "<td class='param'>Subset sample size:</td><td><input class='value' name='sample_size' id='sample_size' value='" + settings[3] + "'></td>"; 
    }
    else if (type == "svm") {
        html += "<td class='param'>Gamma:</td><td><input class='value' name='gamma' id='gamma' value='" + settings[0] + "'></td>"; 
    }
    html += "</tr></table>";
    
    document.getElementById("opts").innerHTML = html;
}

/**
    Returns the default hyperparameter settings for classifier and datasets combinations.
*/
function get_exp_settings(type) {
    if (type == "knn") {
        return [3];
    }
    else if (type == "linear") {
        if (data_name == "iris.csv") return [400, 0.6, 0, 0, 0.1];
        if (data_name == "diabetes.csv") return [500, 0.8, 0, 0, 0.1];
        if (data_name == "spiral.csv") return [100, 0.7, 0, 0, 0.1];
        if (data_name == "glass.csv") return [800, 0.7, 0, 0, 0.1];
        return [1000, 0.5, 0, 0, 0.1];
    }
    else if (type == "nn") {
        if (data_name == "iris.csv") return [8, 400, 0.8, 0, 0.0, 0.1, 0.2];
        if (data_name == "diabetes.csv") return [16, 500, 0.9, 0, 0.0, 0.1, 0.2];
        if (data_name == "spiral.csv") return [72, 1600, 0.8, 0, 0.0, 0.1, 0.2];
        if (data_name == "glass.csv") return [32, 800, 0.7, 0, 0.0, 0.1, 0.2];
        if (data_name == "circle.csv") return ["8,8", 500, 0.5, 0, 0.0, 0.1, 0.2];
        return [32, 500, 0.8, 0, 0.0, 0.1, 0.2];
    }
    else if (type == "dt") {
        return [7, 5];
    }
    else if (type == "rf") {
        if (data_name == "glass.csv") return [25, 7, 5, 0.9];
        return [15, 7, 5, 0.9];
    }
    else if (type == "svm") {
        if (data_name == "iris.csv") return [1];
        if (data_name == "diabetes.csv") return [2];
        if (data_name == "glass.csv") return [1];
        return [5];
    }
}
