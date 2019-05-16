"use strict";

/* Library version */
var VERSION = "0.38";

/** ------------------------------------------------------

Main functions for the Web ML demo.

--------------------------------------------------------- */

/**
    Runs the visualization demo.
*/
async function demo() {
    running = true;
    
    // Wait function
    const delay = ms => new Promise(res => setTimeout(res, ms));
    
    // Disable button
    disable("demo", "running");
    await delay(25);
    
    clear();
    
    // Get classifier type and dataset name
    let type = get_selected_classifier();
    let name = get_selected_dataset();

    // Get render options for the dataset
    let opt = get_render_options(name);
    
    // Shuffle data
    let data = orig_data.clone();
    seed = 42;
    data.shuffle();

    // Build and train classifier
    let cl = build_classifier(type);
    cl.train(data);
    
    if (!cl.iterable()) {
        let map = create_decision_boundaries(cl, opt);
        draw_map(map);
        draw_labels(data, opt);
        calc_accuracy(data, cl);
    }
    else {
        // Enable Stop button
        enable("stop");
        
        let stepsize = get_iteration_stepsize(type, name)
        while (!cl.done() && running) {
            // Start time
            let start = new Date().getTime();

            // Iterate classifier
            cl.iterate_steps(stepsize);
            let map = create_decision_boundaries(cl, opt);
            draw_map(map);
            draw_labels(data, opt);
            calc_accuracy(data, cl);
            document.getElementById("citer").innerHTML = "Iteration: " + cl.current_iteration();
            
            // Time elapsed
            let end = new Date().getTime();
            let time = end - start;
            let rest = 150 - time;
            if (rest < 10) rest = 10;
            // Wait
            await delay(rest);
        }

        // Disable Stop button
        disable("stop");
    }
    
    // Enable button
    enable("demo");

    running = false;
}

/**
    Stops the iteration for iterable classifiers.
*/
function stop_demo() {
    running = false;
}

/**
    Calculates and shows the accuracy on the training dataset.
*/
function calc_accuracy(data, cl) {
    let no_correct = 0;
    for (let i = 0; i < data.no_examples(); i++) {
        let pred = cl.predict([data.x[i]]);
        if (pred == data.y[i]) {
            no_correct++;
        }
    }
    
    let acc = no_correct / data.no_examples() * 100;
    
    let e = document.getElementById("acc").innerHTML = "<font color='black'>Accuracy: " + acc.toFixed(2) + "%</font>&nbsp;&nbsp;(" + no_correct + "/" + data.no_examples() + " correctly classified)";
}

/**
    Creates a map of the dedision boundaries for the current trained classifier.
*/
function create_decision_boundaries(classifier, opt) {
    let map = new Array(100 * 100).fill(0);
    
    // 100x100 map
    for (let x1 = 0; x1 < 100; x1++) {
        for (let x2 = 0; x2 < 100; x2++) {
            // x-values
            let v1 = x1 / 100.0 * opt[0] + opt[2];
            let v2 = x2 / 100.0 * opt[1] + opt[3];
            
            v1 += 0.005;
            v2 += 0.005;
            
            // Prediction
            let pred = classifier.predict([[v1, v2]]);
            // Set predicted label
            map[x1 + x2 * 100] = pred[0];
        }
    }
    return map;
}

/**
    Validates hyperparameter settings.
*/
function validate_setting(id, value, min_val, max_val) {
    let v = value;
    if (v < min_val) {
        v = min_val;
    }
    if (v > max_val) {
        v = max_val;
    }
    if (v != value) {
        document.getElementById(id).value = v;
    }
    return v;
}

/**
    Builds the specified classifier and trains on the specified dataset.
*/
function build_classifier(type) {
    if (type == "knn") {
        // Get options
        let k = get_value("k", 3, 1, 10);
        let dist = get_select_list("dist", 0);

        let cl = new KNN(k, dist);
        return cl;
    }
    else if (type == "nb") {
        let cl = new NaiveBayes();
        return cl;
    }
    else if (type == "nn") {
        // Get options
        let hidden = get_array("hidden", [32]);
        let iter = get_value("iter", 250, 1, 10000);
        let lr = get_value("lr", 0.5, 0.001, 1, parseFloat);
        let batch_size = get_value("batch_size", 0, 0, 1000);
        let l2 = get_value("L2", 0.0, 0, 0.9, parseFloat);
        let momentum = get_value("momentum", 0.1, 0, 0.9, parseFloat);
        let dropout = get_value("dropout", 0.2, 0, 0.9, parseFloat);

        let cl = new NeuralNetwork(hidden, iter, lr, batch_size, 0, l2, momentum, dropout);
        return cl;
    }
    else if (type == "linear") {
        // Get options
        let iter = get_value("iter", 250, 1, 10000);
        let lr = get_value("lr", 0.5, 0.001, 1, parseFloat);
        let batch_size = get_value("batch_size", 0, 0, 1000);
        let l2 = get_value("L2", 0.0, 0, 0.9, parseFloat);
        let momentum = get_value("momentum", 0.1, 0, 0.9, parseFloat);

        let cl = new LinearRegression(iter, lr, batch_size, 0, l2, momentum);
        return cl;
    }
    else if (type == "dt") {
        // Get options
        let maxd = get_value("maxd", 7, 1, 1000);
        let mins = get_value("mins", 5, 1, 1000);
        
        let cl = new CART(maxd, mins);
        return cl;
    }
    else if (type == "rf") {
        // Get options
        let noe = get_value("noe", 11, 1, 500);
        let maxd = get_value("maxd", 7, 1, 1000);
        let mins = get_value("mins", 5, 1, 1000);
        let sample_size = get_value("sample_size", 0.9, 0.1, 1, parseFloat);
        
        let cl = new RandomForest(noe, sample_size, maxd, mins);
        return cl;
    }
    else if (type == "svm") {
        // Get options
        let gamma = get_value("gamma", 40, 0.1, 10000, parseFloat);

        let cl = new SVM(gamma);
        return cl;
    }
    else {
        throw("Unknown classifier: " + type);
    }
}

/** ------------------------------------------------------

Util functions for the VisualML demo.

--------------------------------------------------------- */

// Is set to true if a classifier is running
var running = false;

/**
    Shows the element with the specified id.
*/
function show(id) {
    let e = document.getElementById(id);
    if (e != null) {
        e.style.display = "block";
    }
}

/**
    Hides the element with the specified id.
*/
function hide(id) {
    let e = document.getElementById(id);
    if (e != null) {
        e.style.display = "none";
    }
}

/**
    Toggles visibility of the element with the specified id.
*/
function toggle(id) {
    let e = document.getElementById(id);
    if (e.style.display == "none") {
        e.style.display = "block";
    }
    else {
        e.style.display = "none";
    }
}

/**
    Toggles visibility of the element with the specified id and updates 
    expand arrow.
*/
function toggle_bt(id) {
    let e = document.getElementById(id);
    let bt = document.getElementById(id + "_bt");
    if (e.style.display == "none") {
        e.style.display = "block";
        bt.innerHTML = "&#9660;";
    }
    else {
        e.style.display = "none";
        bt.innerHTML = "&#9658;";
    }
}

/**
    Enables an element.
*/
function enable(id) {
    let e = document.getElementById(id);
    e.disabled = false;
    e.className = "enabled";
}

/**
    Disables an element.
*/
function disable(id, classid="disabled") {
    let e = document.getElementById(id);
    e.disabled = true;
    e.className = classid;
}

/**
    Returns the value of an input field as an integer array.
*/
function get_array(id, default_arr, conv_func=parseInt) {
    let e = document.getElementById(id);
    if (e == null) {
        return default_arr;
    }
    
    let str = e.value;
    
    let arr = str.split(",");
    for (let i in arr) {
        let val = arr[i];
        val = val.trim();
        arr[i] = conv_func(val);
        if (isNaN(arr[i])) {
            e.value = default_arr;
            return default_arr;
        }
    }
    return arr;
}

/**
    Returns the integer value from an input field.
*/
function get_value(id, default_val, min_val, max_val, conv_func=parseInt) {
    // Check if element is available
    let e = document.getElementById(id);
    if (e == null) {
        return default_val;
    }
    
    let str = e.value;
    str = str.trim();
    let val = conv_func(str);
    // Check if valid int
    if (isNaN(val)) {
        e.value = default_val;
        val = default_val;
    }

    // Range check
    if (val < min_val) {
        val = min_val;
        e.value = val;
    }
    if (val > max_val) {
        val = max_val;
        e.value = val;
    }

    return val;
}

/**
    Returns the selected value in a dropdown list.
*/
function get_select_list(id, default_val) {
    // Check if element is available
    let e = document.getElementById(id);
    if (e == null) {
        return default_val;
    }

    // Get selected value
    let val = e.options[e.selectedIndex].value;
    return val;
}


/**
    Returns the selected classifier.
*/
function get_selected_classifier() {
    // Dataset radio buttons
    let name = ""
    let rlist = document.getElementsByName("sel-cl");
    for (let i = 0, length = rlist.length; i < length; i++) {
        if (rlist[i].checked)  {
            name = rlist[i].value;
            break;
        }
    }
    return name;
}

/**
    Returns the selected dataset.
*/
function get_selected_dataset() {
    // Dataset radio buttons
    let name = ""
    let rlist = document.getElementsByName("sel-ds");
    for (let i = 0, length = rlist.length; i < length; i++) {
        if (rlist[i].checked)  {
            name = rlist[i].value;
            break;
        }
    }
    return name;
}

/**
    Shows the settings for the selected classifier.
*/
function update_settings() {
    let type = get_selected_classifier();
    let name = get_selected_dataset();
    
    let settings = get_settings(type, name);
    
    document.getElementById("opts").innerHTML = "";
    
    let html = "<table><tr>";
    if (type == "knn") {
        html += "<td class='param'>Neighbors:</td><td><input class='value' name='k' id='k' value='" + settings[0] + "'></td>";
    }
    else if (type == "nb") {
        html += "<td class='param'>&nbsp;</td>"; 
    }
    else if (type == "linear") {
        html += "<td class='param'>Training iterations:</td><td><input class='value' name='iter' id='iter' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Learning rate:</td><td><input class='value' name='lr' id='lr' value='" + settings[1] + "'></td>";
    }
    else if (type == "nn") {
        html += "<td class='param'>Training iterations:</td><td><input class='value' name='iter' id='iter' value='" + settings[1] + "'></td>";
        html += "<td class='param'>Hidden layers:</td><td><input class='value' name='hidden' id='hidden' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Learning rate:</td><td><input class='value' name='lr' id='lr' value='" + settings[2] + "'></td>";
    }
    else if (type == "dt") {
        html += "<td class='param'>Max depth:</td><td><input class='value' name='maxd' id='maxd' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Min samples split:</td><td><input class='value' name='mins' id='mins' value='" + settings[1] + "'></td>"; 
    }
    else if (type == "rf") {
        html += "<td class='param'>No estimators:</td><td><input class='value' name='noe' id='noe' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Max depth:</td><td><input class='value' name='maxd' id='maxd' value='" + settings[1] + "'></td>";
        html += "<td class='param'>Min samples split:</td><td><input class='value' name='mins' id='mins' value='" + settings[2] + "'></td>"; 
    }
    else if (type == "svm") {
        html += "<td class='param'>Gamma:</td><td><input class='value' name='gamma' id='gamma' value='" + settings[0] + "'></td>"; 
    }
    html += "</tr></table>";
    
    document.getElementById("opts").innerHTML = html;
}

/**
    Shows the dataset labels.
*/
function show_labels() {
    if (running) return;

    clear();
    
    data_name = get_selected_dataset();
    orig_data = get_dataset(data_name);
    let opt = get_render_options(data_name);
    
    draw_labels(orig_data, opt);
}

/**
    Shows the library version in a html "version" field.
*/
function show_version() {
    document.getElementById("version").innerHTML = VERSION;
}


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
        html += "<td class='param' title='Batch size for batch training, or 0 for no batch training'>Batch size:</td><td><input class='value' name='batch_size' id='batch_size' value='" + settings[3] + "'></td>";
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

// Randomizer seed
var seed = 42;
// Original data
var orig_data = null;
// Dataset name
var data_name;

/**
	Custom random function.

	Built-in random has no seed feature. Instead a simple pseudo-random with seed is used.  
	See explanation here:
    https://stackoverflow.com/questions/521295/seeding-the-random-number-generator-in-javascript
*/
function rnd() {
    let x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}

/**
    Random normally distributed using the Box-Muller transform.

    See explanation here:
    https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
*/
function rnd_bm() {
    let u = 0, v = 0;
    while(u === 0) u = rnd(); //Converting [0,1) to (0,1)
    while(v === 0) v = rnd();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) return rnd_bm(); // resample between 0 and 1
    return num;
}

/**
    Dictionary get function.
*/
function get(dict, key) {
    key += "";
    for (let  i = 0; i < dict.length; i++) {
        if (dict[i].key == key) {
            return dict[i].value;
        }
    }
    return null;
}

/** ------------------------------------------------------

Datasets used in the VisualML demo.

--------------------------------------------------------- */

/**
    Spiral dataset.
*/
function get_spiral() {
    let spiral_x = [
	    [ 0.000, 0.000], [ 0.001, 0.010], [ 0.006, 0.019], [ 0.016, 0.026], [ 0.021, 0.035], [ 0.000, 0.051], [ 0.025, 0.055], [ 0.018, 0.068],
		[ 0.024, 0.077], [ 0.039, 0.082], [ 0.042, 0.092], [ 0.075, 0.082], [ 0.072, 0.097], [ 0.069, 0.112], [ 0.086, 0.112], [ 0.094, 0.118],
		[ 0.131, 0.095], [ 0.103, 0.137], [ 0.129, 0.128], [ 0.108, 0.159], [ 0.059, 0.193], [ 0.176, 0.118], [ 0.194, 0.108], [ 0.164, 0.165],
		[ 0.240, 0.036], [ 0.166, 0.190], [ 0.229, 0.128], [ 0.237, 0.135], [ 0.280, 0.037], [ 0.291, 0.031], [ 0.287, 0.098], [ 0.304, 0.075],
		[ 0.290, 0.142], [ 0.269, 0.197], [ 0.331, 0.090], [ 0.351, 0.044], [ 0.361,-0.047], [ 0.369,-0.061], [ 0.381, 0.043], [ 0.393, 0.022],
		[ 0.399, 0.066], [ 0.406, 0.082], [ 0.414, 0.091], [ 0.369,-0.230], [ 0.442,-0.047], [ 0.449,-0.072], [ 0.464,-0.017], [ 0.420,-0.221],
		[ 0.484,-0.022], [ 0.462,-0.177], [ 0.487,-0.135], [ 0.434,-0.277], [ 0.478,-0.218], [ 0.506,-0.176], [ 0.449,-0.310], [ 0.411,-0.373],
		[ 0.431,-0.367], [ 0.404,-0.410], [ 0.468,-0.353], [ 0.440,-0.402], [ 0.456,-0.399], [ 0.419,-0.451], [ 0.449,-0.437], [ 0.514,-0.375],
		[ 0.321,-0.561], [ 0.368,-0.543], [ 0.479,-0.464], [ 0.227,-0.638], [ 0.374,-0.576], [ 0.235,-0.656], [ 0.118,-0.697], [ 0.175,-0.695],
		[ 0.003,-0.727], [ 0.313,-0.667], [ 0.053,-0.746], [ 0.186,-0.734], [ 0.186,-0.745], [ 0.113,-0.769], [ 0.041,-0.787], [-0.049,-0.796],
		[ 0.115,-0.800], [-0.251,-0.779], [-0.217,-0.799], [ 0.080,-0.835], [-0.443,-0.723], [-0.534,-0.672], [-0.468,-0.732], [-0.291,-0.829],
		[-0.176,-0.871], [-0.555,-0.707], [-0.366,-0.832], [-0.646,-0.654], [-0.538,-0.758], [-0.681,-0.647], [-0.632,-0.709], [-0.713,-0.642],
		[-0.653,-0.717], [-0.888,-0.414], [-0.739,-0.658], [-0.807,-0.591], [-0.000,-0.000], [-0.006,-0.008], [-0.013,-0.016], [-0.028,-0.012],
		[-0.029,-0.029], [-0.050,-0.006], [-0.052,-0.032], [-0.059,-0.039], [-0.081,-0.000], [-0.091,-0.005], [-0.101, 0.007], [-0.111,-0.010],
		[-0.112,-0.047], [-0.129, 0.025], [-0.139,-0.028], [-0.151, 0.008], [-0.160, 0.020], [-0.171,-0.010], [-0.180, 0.025], [-0.186, 0.046],
		[-0.199, 0.034], [-0.211,-0.018], [-0.216, 0.052], [-0.206, 0.108], [-0.241, 0.029], [-0.244, 0.067], [-0.254, 0.065], [-0.200, 0.186],
		[-0.241, 0.149], [-0.251, 0.151], [-0.285, 0.103], [-0.250, 0.189], [-0.292, 0.139], [-0.270, 0.196], [-0.296, 0.175], [-0.237, 0.263],
		[-0.238, 0.275], [-0.276, 0.252], [-0.238, 0.301], [-0.315, 0.237], [-0.332, 0.230], [-0.212, 0.355], [-0.223, 0.361], [-0.177, 0.397],
		[-0.013, 0.444], [-0.124, 0.437], [-0.265, 0.382], [-0.076, 0.469], [-0.277, 0.398], [-0.191, 0.457], [-0.138, 0.486], [ 0.062, 0.511],
		[-0.171, 0.497], [-0.162, 0.510], [-0.066, 0.541], [-0.107, 0.545], [ 0.115, 0.554], [-0.112, 0.565], [-0.099, 0.577], [ 0.008, 0.596],
		[ 0.025, 0.606], [ 0.331, 0.520], [ 0.251, 0.574], [ 0.176, 0.612], [ 0.037, 0.645], [ 0.322, 0.572], [ 0.122, 0.655], [ 0.078, 0.672],
		[ 0.443, 0.525], [ 0.375, 0.587], [ 0.471, 0.527], [ 0.434, 0.571], [ 0.520, 0.508], [ 0.377, 0.634], [ 0.358, 0.656], [ 0.586, 0.481],
		[ 0.450, 0.622], [ 0.495, 0.600], [ 0.553, 0.562], [ 0.631, 0.488], [ 0.622, 0.516], [ 0.536, 0.618], [ 0.649, 0.514], [ 0.491, 0.680],
		[ 0.801, 0.279], [ 0.634, 0.579], [ 0.717, 0.491], [ 0.832, 0.283], [ 0.802, 0.384], [ 0.898,-0.045], [ 0.808, 0.417], [ 0.912, 0.114],
		[ 0.920, 0.134], [ 0.889, 0.304], [ 0.948,-0.046], [ 0.958, 0.048], [ 0.954,-0.173], [ 0.954,-0.223], [ 0.850,-0.507], [ 0.916,-0.402],
		[ 0.000,-0.000], [ 0.010,-0.001], [ 0.018,-0.009], [ 0.028,-0.012], [ 0.037,-0.017], [ 0.050,-0.001], [ 0.056,-0.023], [ 0.068,-0.020],
		[ 0.070,-0.041], [ 0.080,-0.043], [ 0.075,-0.068], [ 0.088,-0.068], [ 0.085,-0.086], [ 0.110,-0.072], [ 0.123,-0.071], [ 0.119,-0.094],
		[ 0.114,-0.115], [ 0.106,-0.135], [ 0.044,-0.176], [ 0.119,-0.151], [ 0.146,-0.140], [ 0.128,-0.169], [ 0.131,-0.180], [ 0.090,-0.214],
		[ 0.168,-0.175], [ 0.099,-0.232], [ 0.088,-0.247], [ 0.077,-0.262], [ 0.113,-0.259], [ 0.087,-0.280], [ 0.145,-0.266], [ 0.084,-0.302],
		[ 0.077,-0.314], [ 0.003,-0.333], [ 0.096,-0.330], [-0.051,-0.350], [-0.117,-0.344], [ 0.126,-0.352], [-0.075,-0.377], [-0.111,-0.378],
		[-0.026,-0.403], [-0.063,-0.409], [-0.103,-0.412], [-0.109,-0.421], [-0.128,-0.426], [-0.026,-0.454], [-0.286,-0.366], [-0.302,-0.366],
		[-0.167,-0.455], [-0.128,-0.478], [-0.325,-0.386], [-0.256,-0.447], [-0.340,-0.400], [-0.325,-0.425], [-0.426,-0.341], [-0.447,-0.329],
		[-0.361,-0.435], [-0.326,-0.475], [-0.332,-0.483], [-0.526,-0.280], [-0.418,-0.439], [-0.497,-0.364], [-0.513,-0.359], [-0.570,-0.283],
		[-0.452,-0.462], [-0.621,-0.214], [-0.650,-0.148], [-0.652,-0.181], [-0.654,-0.210], [-0.685,-0.130], [-0.704,-0.062], [-0.557,-0.452],
		[-0.694, 0.218], [-0.737, 0.024], [-0.741,-0.101], [-0.757,-0.033], [-0.756, 0.133], [-0.774, 0.072], [-0.763,-0.195], [-0.654, 0.457],
		[-0.790, 0.172], [-0.725, 0.379], [-0.815, 0.148], [-0.660, 0.517], [-0.762, 0.373], [-0.727, 0.456], [-0.837, 0.232], [-0.636, 0.607],
		[-0.681, 0.571], [-0.586, 0.682], [-0.791, 0.448], [-0.766, 0.508], [-0.352, 0.860], [-0.801, 0.491], [-0.678, 0.665], [-0.461, 0.842],
		[-0.601, 0.761], [-0.496, 0.845], [-0.628, 0.766], [-0.473, 0.881]
	];
    let spiral_y = [
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    	2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    	2,2,2,2,2
    ];
    
    let data = new Dataset();
    data.set(spiral_x, spiral_y);
    return data;
}

/**
	Circle dataset.
*/
function get_circle() {
	let circle_x = [
		[0.564,0.542], [0.478,0.518], [0.431,0.599], [0.591,0.386], [0.401,0.505], [0.605,0.494], [0.422,0.442], [0.433,0.264],
		[0.254,0.358], [0.438,0.579], [0.418,0.273], [0.562,0.446], [0.565,0.344], [0.480,0.475], [0.477,0.506], [0.671,0.387],
		[0.483,0.360], [0.395,0.448], [0.509,0.434], [0.538,0.472], [0.530,0.425], [0.264,0.392], [0.402,0.495], [0.520,0.423],
		[0.526,0.622], [0.470,0.540], [0.259,0.587], [0.607,0.563], [0.494,0.516], [0.628,0.457], [0.527,0.492], [0.516,0.537],
		[0.672,0.534], [0.506,0.255], [0.540,0.432], [0.440,0.389], [0.385,0.718], [0.494,0.538], [0.218,0.459], [0.235,0.612],
		[0.487,0.572], [0.493,0.551], [0.495,0.589], [0.448,0.318], [0.588,0.323], [0.460,0.319], [0.400,0.246], [0.646,0.543],
		[0.401,0.379], [0.417,0.280], [0.480,0.335], [0.317,0.520], [0.451,0.516], [0.511,0.462], [0.493,0.370], [0.568,0.542],
		[0.393,0.555], [0.539,0.197], [0.534,0.415], [0.466,0.517], [0.326,0.423], [0.638,0.634], [0.464,0.359], [0.406,0.432],
		[0.415,0.555], [0.477,0.453], [0.473,0.502], [0.435,0.334], [0.458,0.547], [0.403,0.364], [0.630,0.522], [0.637,0.468],
		[0.318,0.490], [0.454,0.491], [0.543,0.494], [0.565,0.508], [0.331,0.608], [0.370,0.520], [0.422,0.309], [0.406,0.489],
		[0.419,0.508], [0.444,0.548], [0.430,0.337], [0.377,0.360], [0.397,0.467], [0.507,0.604], [0.379,0.571], [0.482,0.546],
		[0.506,0.530], [0.400,0.475], [0.523,0.270], [0.447,0.611], [0.348,0.546], [0.415,0.367], [0.370,0.595], [0.417,0.625],
		[0.368,0.389], [0.369,0.246], [0.513,0.367], [0.287,0.337], [0.548,0.458], [0.410,0.457], [0.336,0.458], [0.486,0.487],
		[0.488,0.452], [0.581,0.551], [0.529,0.426], [0.485,0.349], [0.366,0.320], [0.649,0.588], [0.576,0.451], [0.506,0.527],
		[0.404,0.450], [0.569,0.609], [0.431,0.524], [0.489,0.546], [0.523,0.521], [0.545,0.604], [0.487,0.348], [0.362,0.632],
		[0.523,0.417], [0.525,0.460], [0.246,0.318], [0.555,0.416], [0.495,0.536], [0.510,0.582], [0.243,0.395], [0.453,0.539],
		[0.407,0.359], [0.522,0.402], [0.472,0.498], [0.286,0.509], [0.341,0.509], [0.574,0.423], [0.273,0.310], [0.459,0.313],
		[0.425,0.389], [0.339,0.318], [0.451,0.401], [0.352,0.603], [0.602,0.338], [0.487,0.291], [0.444,0.446], [0.476,0.481],
		[0.540,0.408], [0.521,0.402], [0.285,0.429], [0.396,0.478], [0.525,0.578], [0.477,0.393], [0.321,0.894], [0.384,0.101],
		[0.678,0.868], [0.864,0.205], [0.751,0.208], [0.101,0.307], [0.274,0.136], [0.921,0.964], [0.076,0.955], [0.760,0.911],
		[0.912,0.753], [0.707,0.840], [0.919,0.642], [0.495,0.871], [0.356,0.970], [0.091,0.773], [0.970,0.221], [0.348,0.911],
		[0.748,0.759], [0.219,0.838], [0.128,0.535], [0.296,0.095], [0.696,0.909], [0.938,0.520], [0.831,0.643], [0.581,0.845],
		[0.176,0.118], [0.201,0.042], [0.078,0.233], [0.891,0.772], [0.161,0.856], [0.532,0.144], [0.424,0.952], [0.089,0.725],
		[0.215,0.044], [0.049,0.235], [0.511,0.879], [0.174,0.636], [0.913,0.075], [0.135,0.154], [0.188,0.932], [0.792,0.214],
		[0.126,0.663], [0.179,0.731], [0.580,0.053], [0.317,0.815], [0.800,0.264], [0.195,0.024], [0.615,0.851], [0.253,0.086],
		[0.833,0.318], [0.670,0.920], [0.107,0.109], [0.899,0.204], [0.295,0.035], [0.085,0.197], [0.895,0.704], [0.172,0.218],
		[0.954,0.487], [0.127,0.313], [0.507,0.971], [0.164,0.188], [0.690,0.817], [0.076,0.350], [0.781,0.216], [0.707,0.813],
		[0.620,0.075], [0.035,0.216], [0.450,0.029], [0.246,0.205], [0.351,0.840], [0.524,0.938], [0.108,0.257], [0.722,0.197],
		[0.479,0.931], [0.274,0.127], [0.963,0.161], [0.450,0.112], [0.128,0.883], [0.234,0.779], [0.259,0.214], [0.322,0.047],
		[0.316,0.021], [0.952,0.394], [0.187,0.214], [0.771,0.191], [0.217,0.097], [0.976,0.326], [0.150,0.156], [0.235,0.202],
		[0.924,0.322], [0.476,0.075], [0.305,0.863], [0.467,0.104], [0.614,0.086], [0.973,0.467], [0.372,0.887], [0.683,0.090],
		[0.447,0.143], [0.308,0.055], [0.850,0.931], [0.203,0.113], [0.900,0.688], [0.266,0.934], [0.866,0.308], [0.743,0.039],
		[0.523,0.047], [0.270,0.766], [0.049,0.542], [0.623,0.834], [0.249,0.135], [0.416,0.863], [0.587,0.093], [0.976,0.392],
		[0.253,0.864], [0.915,0.324], [0.118,0.288], [0.106,0.310], [0.703,0.209], [0.307,0.077], [0.786,0.926], [0.053,0.801],
		[0.392,0.065], [0.875,0.497], [0.028,0.658], [0.897,0.851], [0.805,0.123], [0.954,0.920], [0.340,0.977], [0.270,0.100],
		[0.053,0.782], [0.266,0.949], [0.712,0.166], [0.170,0.730], [0.646,0.836], [0.148,0.267], [0.140,0.606], [0.039,0.630],
		[0.162,0.297], [0.966,0.760], [0.879,0.264], [0.862,0.770], [0.955,0.838], [0.797,0.782], [0.046,0.422], [0.117,0.460],
		[0.903,0.735], [0.968,0.494], [0.704,0.063], [0.073,0.350]
	];
	let circle_y = [
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1
	];

	let data = new Dataset();
    data.set(circle_x, circle_y);
    return data;
}

/**
    Flame dataset.
*/
function get_flame() {
    let flame_x = [
		[0.123,0.273], [0.090,0.312], [0.093,0.425], [0.057,0.432], [0.033,0.455], [0.043,0.488], [0.073,0.465], [0.090,0.445],
		[0.130,0.440], [0.160,0.452], [0.120,0.467], [0.167,0.472], [0.197,0.487], [0.127,0.492], [0.090,0.485], [0.090,0.503],
		[0.083,0.522], [0.117,0.532], [0.133,0.513], [0.167,0.500], [0.113,0.565], [0.160,0.532], [0.203,0.518], [0.247,0.518],
		[0.230,0.537], [0.197,0.550], [0.160,0.553], [0.160,0.592], [0.190,0.575], [0.217,0.565], [0.263,0.547], [0.180,0.607],
		[0.230,0.598], [0.253,0.582], [0.267,0.563], [0.297,0.537], [0.310,0.562], [0.323,0.585], [0.287,0.598], [0.223,0.623],
		[0.247,0.657], [0.293,0.635], [0.283,0.620], [0.320,0.612], [0.350,0.592], [0.383,0.582], [0.353,0.558], [0.403,0.548],
		[0.433,0.570], [0.403,0.593], [0.373,0.607], [0.363,0.628], [0.337,0.648], [0.303,0.665], [0.330,0.685], [0.390,0.707],
		[0.373,0.690], [0.377,0.667], [0.397,0.640], [0.417,0.653], [0.407,0.618], [0.440,0.612], [0.443,0.590], [0.487,0.588],
		[0.523,0.590], [0.477,0.607], [0.507,0.610], [0.447,0.625], [0.487,0.625], [0.447,0.640], [0.487,0.645], [0.450,0.657],
		[0.493,0.660], [0.437,0.675], [0.490,0.673], [0.453,0.702], [0.497,0.697], [0.457,0.718], [0.507,0.713], [0.570,0.712],
		[0.547,0.683], [0.527,0.663], [0.537,0.650], [0.520,0.633], [0.533,0.618], [0.560,0.597], [0.577,0.608], [0.593,0.630],
		[0.560,0.630], [0.577,0.645], [0.563,0.665], [0.590,0.688], [0.640,0.690], [0.610,0.667], [0.680,0.667], [0.633,0.645],
		[0.717,0.647], [0.697,0.627], [0.657,0.630], [0.627,0.613], [0.677,0.610], [0.657,0.595], [0.603,0.592], [0.620,0.577],
		[0.610,0.562], [0.567,0.573], [0.777,0.618], [0.740,0.612], [0.693,0.592], [0.667,0.568], [0.797,0.592], [0.750,0.587],
		[0.707,0.570], [0.743,0.567], [0.793,0.572], [0.840,0.570], [0.787,0.552], [0.737,0.552], [0.687,0.553], [0.660,0.542],
		[0.697,0.533], [0.870,0.537], [0.833,0.542], [0.793,0.532], [0.747,0.525], [0.723,0.505], [0.760,0.492], [0.780,0.513],
		[0.820,0.518], [0.863,0.515], [0.837,0.502], [0.803,0.492], [0.783,0.463], [0.817,0.472], [0.853,0.483], [0.903,0.500],
		[0.907,0.480], [0.863,0.467], [0.833,0.458], [0.813,0.438], [0.847,0.422], [0.867,0.443], [0.903,0.460], [0.937,0.458],
		[0.947,0.432], [0.940,0.413], [0.900,0.440], [0.890,0.417], [0.887,0.400], [0.487,0.562], [0.530,0.555], [0.513,0.532],
		[0.450,0.537], [0.350,0.522], [0.410,0.510], [0.467,0.510], [0.507,0.493], [0.570,0.513], [0.623,0.517], [0.553,0.485],
		[0.527,0.480], [0.477,0.475], [0.447,0.490], [0.347,0.497], [0.413,0.468], [0.450,0.453], [0.410,0.450], [0.377,0.460],
		[0.310,0.448], [0.273,0.418], [0.357,0.440], [0.493,0.447], [0.517,0.463], [0.567,0.457], [0.620,0.467], [0.647,0.435],
		[0.587,0.435], [0.537,0.437], [0.507,0.428], [0.457,0.433], [0.413,0.425], [0.380,0.420], [0.340,0.415], [0.303,0.395],
		[0.367,0.400], [0.407,0.398], [0.433,0.413], [0.450,0.402], [0.487,0.408], [0.553,0.420], [0.593,0.410], [0.637,0.412],
		[0.690,0.397], [0.530,0.398], [0.263,0.387], [0.250,0.358], [0.260,0.335], [0.303,0.312], [0.350,0.308], [0.433,0.280],
		[0.497,0.280], [0.557,0.288], [0.617,0.293], [0.663,0.317], [0.703,0.347], [0.660,0.368], [0.613,0.383], [0.570,0.393],
		[0.587,0.373], [0.613,0.355], [0.637,0.332], [0.603,0.313], [0.587,0.340], [0.543,0.322], [0.537,0.340], [0.557,0.360],
		[0.527,0.357], [0.537,0.377], [0.487,0.387], [0.503,0.372], [0.457,0.385], [0.417,0.378], [0.370,0.383], [0.310,0.363],
		[0.333,0.348], [0.370,0.330], [0.370,0.358], [0.413,0.360], [0.453,0.365], [0.493,0.358], [0.443,0.352], [0.410,0.340],
		[0.433,0.330], [0.440,0.313], [0.513,0.312], [0.500,0.327], [0.500,0.345], [0.470,0.338], [0.460,0.295], [0.410,0.303]
    ];
    let flame_y = [
    	0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0
    ];
    
    let data = new Dataset();
    data.set(flame_x, flame_y);
    return data;
}

/**
    Moons dataset.
*/
function get_moons() {
    let moons_x = [
		[0.019,0.412], [0.017,0.453], [0.073,0.457], [0.117,0.484], [0.109,0.452], [0.119,0.448], [0.113,0.402], [0.102,0.394],
		[0.090,0.383], [0.076,0.362], [0.064,0.330], [0.069,0.314], [0.087,0.314], [0.098,0.354], [0.160,0.478], [0.170,0.433],
		[0.158,0.386], [0.157,0.358], [0.130,0.343], [0.122,0.316], [0.146,0.316], [0.134,0.304], [0.116,0.280], [0.101,0.269],
		[0.113,0.258], [0.180,0.214], [0.226,0.184], [0.217,0.233], [0.204,0.331], [0.249,0.293], [0.280,0.287], [0.294,0.278],
		[0.259,0.203], [0.277,0.188], [0.296,0.181], [0.304,0.183], [0.314,0.202], [0.312,0.210], [0.337,0.262], [0.338,0.250],
		[0.271,0.336], [0.270,0.323], [0.283,0.310], [0.292,0.314], [0.306,0.311], [0.310,0.296], [0.320,0.297], [0.316,0.308],
		[0.313,0.317], [0.312,0.324], [0.382,0.249], [0.393,0.248], [0.390,0.240], [0.378,0.203], [0.368,0.198], [0.426,0.237],
		[0.418,0.251], [0.476,0.226], [0.351,0.326], [0.369,0.330], [0.388,0.339], [0.400,0.334], [0.406,0.351], [0.400,0.304],
		[0.413,0.306], [0.427,0.312], [0.432,0.309], [0.447,0.320], [0.447,0.336], [0.442,0.348], [0.432,0.377], [0.428,0.384],
		[0.473,0.304], [0.509,0.274], [0.514,0.264], [0.539,0.292], [0.490,0.350], [0.466,0.394], [0.481,0.417], [0.479,0.429],
		[0.480,0.438], [0.478,0.456], [0.498,0.433], [0.494,0.398], [0.514,0.377], [0.522,0.360], [0.528,0.351], [0.559,0.360],
		[0.567,0.368], [0.511,0.400], [0.532,0.406], [0.576,0.410], [0.614,0.452], [0.513,0.476], [0.522,0.462], [0.534,0.469],
		[0.544,0.473], [0.314,0.414], [0.318,0.427], [0.318,0.450], [0.328,0.464], [0.341,0.456], [0.354,0.434], [0.367,0.421],
		[0.386,0.421], [0.381,0.438], [0.370,0.442], [0.367,0.463], [0.361,0.468], [0.356,0.483], [0.353,0.507], [0.337,0.532],
		[0.338,0.540], [0.378,0.452], [0.376,0.459], [0.386,0.457], [0.381,0.464], [0.384,0.469], [0.393,0.467], [0.378,0.476],
		[0.374,0.482], [0.369,0.488], [0.380,0.489], [0.388,0.486], [0.396,0.484], [0.391,0.492], [0.382,0.500], [0.383,0.508],
		[0.380,0.517], [0.377,0.526], [0.367,0.529], [0.361,0.522], [0.357,0.536], [0.370,0.558], [0.371,0.547], [0.377,0.550],
		[0.384,0.551], [0.401,0.536], [0.413,0.522], [0.420,0.532], [0.416,0.550], [0.399,0.558], [0.409,0.577], [0.388,0.569],
		[0.391,0.574], [0.393,0.581], [0.384,0.584], [0.377,0.584], [0.372,0.586], [0.440,0.579], [0.424,0.588], [0.389,0.616],
		[0.390,0.620], [0.397,0.632], [0.404,0.614], [0.429,0.598], [0.431,0.603], [0.423,0.603], [0.420,0.611], [0.413,0.626],
		[0.416,0.630], [0.430,0.618], [0.443,0.616], [0.444,0.602], [0.451,0.602], [0.457,0.604], [0.408,0.646], [0.414,0.647],
		[0.429,0.644], [0.424,0.648], [0.426,0.652], [0.471,0.604], [0.476,0.604], [0.469,0.622], [0.453,0.644], [0.456,0.659],
		[0.447,0.666], [0.454,0.686], [0.466,0.677], [0.466,0.662], [0.464,0.653], [0.468,0.644], [0.486,0.611], [0.487,0.618],
		[0.496,0.629], [0.486,0.652], [0.473,0.688], [0.502,0.651], [0.500,0.663], [0.526,0.640], [0.536,0.644], [0.488,0.693],
		[0.492,0.688], [0.499,0.682], [0.499,0.691], [0.504,0.678], [0.511,0.676], [0.516,0.682], [0.521,0.668], [0.528,0.668],
		[0.543,0.663], [0.547,0.657], [0.560,0.654], [0.579,0.658], [0.562,0.672], [0.541,0.681], [0.518,0.691], [0.510,0.694],
		[0.498,0.699], [0.507,0.709], [0.509,0.711], [0.517,0.714], [0.521,0.720], [0.523,0.707], [0.529,0.719], [0.529,0.694],
		[0.538,0.711], [0.546,0.711], [0.549,0.714], [0.549,0.704], [0.553,0.694], [0.587,0.673], [0.603,0.668], [0.607,0.679],
		[0.611,0.679], [0.612,0.687], [0.597,0.690], [0.591,0.691], [0.597,0.702], [0.582,0.702], [0.578,0.706], [0.559,0.709],
		[0.569,0.713], [0.574,0.720], [0.554,0.726], [0.558,0.728], [0.566,0.730], [0.597,0.734], [0.603,0.730], [0.604,0.733],
		[0.621,0.728], [0.621,0.722], [0.640,0.710], [0.640,0.696], [0.639,0.679], [0.636,0.672], [0.650,0.660], [0.667,0.654],
		[0.680,0.724], [0.668,0.723], [0.661,0.723], [0.649,0.711], [0.654,0.710], [0.646,0.699], [0.653,0.692], [0.656,0.696],
		[0.664,0.701], [0.683,0.701], [0.676,0.710], [0.684,0.712], [0.690,0.712], [0.687,0.684], [0.681,0.670], [0.682,0.663],
		[0.700,0.661], [0.703,0.654], [0.711,0.644], [0.722,0.623], [0.741,0.634], [0.724,0.646], [0.726,0.653], [0.723,0.659],
		[0.719,0.664], [0.723,0.671], [0.716,0.688], [0.719,0.706], [0.731,0.708], [0.727,0.698], [0.728,0.692], [0.758,0.698],
		[0.758,0.689], [0.747,0.683], [0.741,0.674], [0.750,0.668], [0.742,0.662], [0.766,0.671], [0.770,0.674], [0.770,0.661],
		[0.783,0.661], [0.763,0.649], [0.758,0.641], [0.766,0.638], [0.771,0.640], [0.774,0.644], [0.763,0.628], [0.768,0.626],
		[0.779,0.622], [0.789,0.621], [0.796,0.642], [0.813,0.651], [0.817,0.639], [0.811,0.636], [0.799,0.624], [0.802,0.620],
		[0.803,0.613], [0.836,0.637], [0.842,0.630], [0.648,0.702], [0.776,0.600], [0.784,0.591], [0.798,0.592], [0.800,0.586],
		[0.794,0.578], [0.816,0.597], [0.813,0.582], [0.820,0.583], [0.828,0.574], [0.809,0.574], [0.807,0.562], [0.817,0.559],
		[0.848,0.584], [0.853,0.590], [0.852,0.567], [0.838,0.560], [0.832,0.552], [0.830,0.547], [0.822,0.539], [0.818,0.529],
		[0.826,0.521], [0.828,0.530], [0.837,0.534], [0.843,0.537], [0.858,0.539], [0.856,0.529], [0.844,0.512], [0.829,0.511],
		[0.833,0.502], [0.841,0.478], [0.851,0.476], [0.846,0.479], [0.852,0.481], [0.856,0.483], [0.873,0.484], [0.867,0.507],
		[0.866,0.513], [0.871,0.526], [0.878,0.538], [0.879,0.527], [0.883,0.517], [0.893,0.516], [0.898,0.532], [0.899,0.522],
		[0.901,0.508], [0.899,0.478], [0.893,0.471], [0.903,0.469], [0.902,0.461], [0.918,0.460], [0.910,0.451], [0.917,0.427],
		[0.910,0.421], [0.904,0.434], [0.899,0.438], [0.887,0.440], [0.881,0.440], [0.872,0.456], [0.863,0.456], [0.851,0.433],
		[0.861,0.426], [0.867,0.431], [0.850,0.414], [0.878,0.423], [0.887,0.421]
    ];
    let moons_y = [
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ];
    
    let data = new Dataset();
    data.set(moons_x, moons_y);
    return data;
}

/**
    Gaussian dataset.
*/
function get_gaussian() {
	let gaussian_x = [
		[0.351,0.334], [0.282,0.315], [0.245,0.379], [0.373,0.209], [0.221,0.304], [0.384,0.295], [0.238,0.253], [0.246,0.111],
		[0.103,0.187], [0.250,0.363], [0.234,0.119], [0.349,0.257], [0.352,0.175], [0.284,0.280], [0.282,0.305], [0.437,0.209],
		[0.286,0.188], [0.216,0.258], [0.308,0.248], [0.331,0.278], [0.324,0.240], [0.111,0.214], [0.222,0.296], [0.316,0.238],
		[0.321,0.398], [0.276,0.332], [0.107,0.369], [0.386,0.350], [0.295,0.313], [0.402,0.265], [0.322,0.293], [0.313,0.329],
		[0.438,0.327], [0.305,0.104], [0.332,0.245], [0.252,0.211], [0.208,0.474], [0.295,0.330], [0.075,0.268], [0.088,0.389],
		[0.290,0.358], [0.295,0.341], [0.296,0.371], [0.259,0.155], [0.370,0.158], [0.268,0.155], [0.220,0.097], [0.417,0.334],
		[0.221,0.203], [0.233,0.124], [0.284,0.168], [0.154,0.316], [0.261,0.313], [0.309,0.270], [0.295,0.196], [0.354,0.334],
		[0.215,0.344], [0.331,0.057], [0.327,0.232], [0.272,0.313], [0.161,0.239], [0.411,0.407], [0.271,0.187], [0.225,0.246],
		[0.232,0.344], [0.281,0.262], [0.279,0.302], [0.248,0.167], [0.266,0.338], [0.222,0.192], [0.404,0.318], [0.410,0.274],
		[0.155,0.292], [0.263,0.293], [0.334,0.295], [0.352,0.307], [0.165,0.386], [0.196,0.316], [0.238,0.147], [0.225,0.291],
		[0.235,0.307], [0.255,0.339], [0.244,0.170], [0.202,0.188], [0.217,0.273], [0.306,0.383], [0.203,0.356], [0.285,0.337],
		[0.305,0.324], [0.220,0.280], [0.318,0.116], [0.257,0.389], [0.178,0.337], [0.232,0.194], [0.196,0.376], [0.234,0.400],
		[0.194,0.211], [0.195,0.097], [0.310,0.194], [0.129,0.170], [0.338,0.266], [0.228,0.265], [0.169,0.267], [0.289,0.289],
		[0.291,0.261], [0.365,0.341], [0.324,0.241], [0.288,0.179], [0.193,0.156], [0.419,0.371], [0.361,0.261], [0.305,0.322],
		[0.223,0.260], [0.355,0.387], [0.245,0.319], [0.291,0.336], [0.318,0.317], [0.336,0.383], [0.289,0.178], [0.190,0.405],
		[0.318,0.234], [0.320,0.268], [0.097,0.154], [0.344,0.233], [0.296,0.329], [0.708,0.766], [0.495,0.616], [0.662,0.731],
		[0.626,0.587], [0.718,0.621], [0.678,0.698], [0.529,0.707], [0.573,0.707], [0.759,0.639], [0.519,0.548], [0.667,0.550],
		[0.640,0.611], [0.571,0.555], [0.661,0.621], [0.581,0.782], [0.782,0.571], [0.690,0.533], [0.656,0.657], [0.681,0.685],
		[0.732,0.627], [0.717,0.621], [0.528,0.643], [0.617,0.682], [0.720,0.762], [0.682,0.614], [0.635,0.715], [0.641,0.594],
		[0.779,0.754], [0.693,0.556], [0.733,0.526], [0.692,0.634], [0.713,0.598], [0.633,0.768], [0.696,0.720], [0.591,0.752],
		[0.616,0.473], [0.791,0.643], [0.675,0.683], [0.596,0.779], [0.745,0.537], [0.659,0.552], [0.551,0.635], [0.587,0.738],
		[0.700,0.725], [0.713,0.678], [0.659,0.747], [0.605,0.523], [0.625,0.820], [0.572,0.585], [0.554,0.709], [0.643,0.671],
		[0.640,0.715], [0.724,0.727], [0.688,0.741], [0.572,0.642], [0.582,0.588], [0.669,0.701], [0.740,0.685], [0.682,0.705],
		[0.636,0.808], [0.723,0.794], [0.591,0.504], [0.744,0.696], [0.682,0.751], [0.612,0.678], [0.682,0.575], [0.655,0.657],
		[0.528,0.698], [0.689,0.680], [0.598,0.572], [0.492,0.747], [0.668,0.568], [0.614,0.628], [0.553,0.639], [0.755,0.777],
		[0.609,0.658], [0.663,0.599], [0.742,0.684], [0.662,0.744], [0.573,0.696], [0.762,0.591], [0.676,0.683], [0.505,0.735],
		[0.697,0.693], [0.594,0.689], [0.552,0.751], [0.650,0.799], [0.652,0.606], [0.615,0.738], [0.812,0.731], [0.682,0.513],
		[0.725,0.608], [0.602,0.627], [0.757,0.681], [0.759,0.681], [0.650,0.726], [0.613,0.599], [0.654,0.544], [0.772,0.671],
		[0.582,0.763], [0.657,0.431], [0.670,0.653], [0.617,0.523], [0.491,0.701], [0.709,0.685], [0.614,0.620], [0.709,0.659],
		[0.730,0.643], [0.699,0.412], [0.592,0.813], [0.588,0.589], [0.626,0.628], [0.786,0.731], [0.540,0.611], [0.611,0.643],
		[0.628,0.647], [0.584,0.429], [0.713,0.606], [0.707,0.447], [0.719,0.587], [0.675,0.606], [0.857,0.683], [0.719,0.805],
		[0.656,0.623], [0.622,0.685], [0.080,0.590], [0.163,0.596], [0.202,0.620], [0.182,0.794], [0.363,0.430], [0.303,0.739],
		[0.246,0.727], [0.311,0.741], [0.237,0.552], [0.407,0.621], [0.179,0.724], [0.180,0.568], [0.224,0.574], [0.063,0.689],
		[0.265,0.649], [0.312,0.761], [0.284,0.881], [0.387,0.677], [0.303,0.638], [0.221,0.675], [0.216,0.623], [0.281,0.703],
		[0.236,0.630], [0.380,0.620], [0.264,0.496], [0.358,0.657], [0.100,0.584], [0.282,0.654], [0.125,0.665], [0.270,0.629],
		[0.243,0.770], [0.306,0.722], [0.310,0.768], [0.208,0.727], [0.343,0.448], [0.287,0.608], [0.271,0.575], [0.186,0.637],
		[0.464,0.625], [0.222,0.736], [0.277,0.590], [0.424,0.788], [0.233,0.537], [0.413,0.550], [0.285,0.675], [0.231,0.625],
		[0.332,0.519], [0.234,0.756], [0.445,0.698], [0.314,0.680], [0.231,0.714], [0.110,0.382], [0.374,0.548], [0.328,0.609],
		[0.359,0.536], [0.113,0.648], [0.338,0.718], [0.298,0.504], [0.272,0.467], [0.172,0.740], [0.188,0.656], [0.413,0.817],
		[0.169,0.640], [0.218,0.642], [0.118,0.663], [0.246,0.716], [0.318,0.518], [0.418,0.579], [0.298,0.603], [0.408,0.681],
		[0.452,0.574], [0.210,0.709], [0.309,0.625], [0.276,0.741], [0.184,0.618], [0.334,0.776], [0.307,0.683], [0.416,0.647],
		[0.228,0.601], [0.196,0.728], [0.234,0.586], [0.357,0.649], [0.357,0.733], [0.237,0.654], [0.139,0.806], [0.123,0.578],
		[0.287,0.608], [0.205,0.525], [0.272,0.623], [0.493,0.616], [0.001,0.746], [0.151,0.816], [0.283,0.641], [0.080,0.679],
		[0.333,0.566], [0.302,0.568], [0.232,0.510], [0.289,0.465], [0.068,0.392], [0.294,0.753], [0.311,0.604], [0.171,0.450],
		[0.246,0.522], [0.272,0.409], [0.147,0.680], [0.240,0.709], [0.291,0.744], [0.367,0.849], [0.097,0.707], [0.407,0.626],
		[0.307,0.614], [0.337,0.958], [0.411,0.439], [0.395,0.655], [0.182,0.776], [0.315,0.712], [0.035,0.654], [0.139,0.660],
		[0.148,0.639], [0.355,0.472], [0.295,0.676], [0.224,0.684], [0.057,0.381], [0.150,0.567], [0.266,0.616], [0.591,0.243],
		[0.714,0.310], [0.735,0.106], [0.562,0.270], [0.614,0.385], [0.574,0.236], [0.619,0.330], [0.640,0.360], [0.628,0.287],
		[0.605,0.271], [0.690,0.206], [0.528,0.306], [0.583,0.163], [0.680,0.264], [0.467,0.250], [0.596,0.335], [0.609,0.190],
		[0.672,0.371], [0.669,0.277], [0.779,0.256], [0.669,0.398], [0.563,0.331], [0.547,0.265], [0.564,0.213], [0.627,0.302],
		[0.602,0.251], [0.512,0.190], [0.410,0.188], [0.619,0.294], [0.574,0.274], [0.614,0.305], [0.546,0.313], [0.639,0.189],
		[0.730,0.320], [0.633,0.300], [0.740,0.353], [0.972,0.274], [0.712,0.398], [0.604,0.250], [0.792,0.166], [0.627,0.276],
		[0.818,0.278], [0.624,0.297], [0.483,0.335], [0.537,0.253], [0.602,0.226], [0.763,0.251], [0.549,0.334], [0.790,0.237],
		[0.771,0.301], [0.678,0.203], [0.628,0.171], [0.615,0.208], [0.535,0.145], [0.730,0.262], [0.442,0.180], [0.605,0.218],
		[0.558,0.367], [0.800,0.220], [0.702,0.305], [0.675,0.162], [0.443,0.226], [0.659,0.311], [0.715,0.392], [0.556,0.107],
		[0.580,0.177], [0.635,0.272], [0.709,0.342], [0.640,0.237], [0.522,0.366], [0.416,0.284], [0.710,0.317], [0.598,0.166],
		[0.656,0.088], [0.641,0.204], [0.627,0.226], [0.627,0.139], [0.705,0.248], [0.709,0.290], [0.565,0.382], [0.485,0.410],
		[0.793,0.366], [0.680,0.371], [0.544,0.205], [0.711,0.306], [0.691,0.233], [0.693,0.283], [0.726,0.295], [0.915,0.283],
		[0.594,0.174], [0.641,0.179], [0.656,0.250], [0.706,0.365], [0.815,0.273], [0.648,0.311], [0.608,0.352], [0.760,0.314],
		[0.799,0.322], [0.640,0.235], [0.643,0.165], [0.581,0.353], [0.790,0.234], [0.656,0.406], [0.627,0.361], [0.730,0.364],
		[0.547,0.364], [0.519,0.358], [0.614,0.143], [0.493,0.135], [0.457,0.188], [0.768,0.354], [0.692,0.243], [0.677,0.164],
		[0.734,0.323], [0.819,0.288], [0.792,0.251], [0.670,0.284], [0.591,0.243], [0.769,0.191], [0.597,0.186], [0.704,0.301],
		[0.484,0.111], [0.618,0.156], [0.714,0.281], [0.695,0.351]
	];
	let gaussian_y = [
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
	];

	let data = new Dataset();
    data.set(gaussian_x, gaussian_y);
    return data;
}

/**
    Aggregation dataset.
*/
function get_aggregation() {
    let agg_x = [
		[0.389,0.716], [0.361,0.709], [0.344,0.701], [0.325,0.729], [0.340,0.662], [0.310,0.696], [0.305,0.716], [0.324,0.649],
		[0.296,0.675], [0.279,0.718], [0.269,0.693], [0.241,0.711], [0.269,0.664], [0.290,0.647], [0.315,0.601], [0.277,0.630],
		[0.251,0.649], [0.233,0.681], [0.188,0.706], [0.212,0.676], [0.196,0.670], [0.170,0.671], [0.189,0.657], [0.225,0.646],
		[0.235,0.639], [0.221,0.615], [0.264,0.609], [0.264,0.589], [0.230,0.597], [0.184,0.619], [0.165,0.644], [0.145,0.674],
		[0.133,0.653], [0.135,0.631], [0.120,0.626], [0.160,0.620], [0.185,0.606], [0.107,0.600], [0.084,0.583], [0.107,0.569],
		[0.148,0.589], [0.171,0.581], [0.174,0.564], [0.138,0.565], [0.095,0.546], [0.104,0.509], [0.140,0.519], [0.160,0.549],
		[0.186,0.549], [0.194,0.530], [0.174,0.495], [0.151,0.505], [0.134,0.476], [0.158,0.478], [0.179,0.447], [0.205,0.501],
		[0.208,0.463], [0.226,0.455], [0.223,0.441], [0.251,0.430], [0.215,0.522], [0.216,0.547], [0.224,0.570], [0.224,0.555],
		[0.264,0.557], [0.284,0.586], [0.307,0.569], [0.279,0.551], [0.271,0.526], [0.246,0.517], [0.231,0.491], [0.267,0.509],
		[0.309,0.540], [0.321,0.519], [0.275,0.496], [0.247,0.466], [0.285,0.458], [0.309,0.470], [0.304,0.453], [0.299,0.431],
		[0.326,0.435], [0.338,0.466], [0.350,0.497], [0.388,0.429], [0.344,0.415], [0.195,0.342], [0.225,0.318], [0.201,0.323],
		[0.170,0.330], [0.155,0.314], [0.143,0.306], [0.129,0.284], [0.152,0.294], [0.176,0.311], [0.214,0.302], [0.177,0.299],
		[0.171,0.273], [0.148,0.258], [0.176,0.251], [0.191,0.277], [0.220,0.285], [0.225,0.273], [0.254,0.275], [0.273,0.250],
		[0.294,0.271], [0.276,0.227], [0.321,0.266], [0.340,0.277], [0.362,0.295], [0.425,0.323], [0.396,0.300], [0.406,0.292],
		[0.370,0.284], [0.344,0.261], [0.329,0.245], [0.295,0.224], [0.259,0.193], [0.283,0.199], [0.328,0.224], [0.352,0.251],
		[0.284,0.173], [0.310,0.177], [0.324,0.190], [0.334,0.206], [0.358,0.233], [0.378,0.256], [0.409,0.271], [0.406,0.255],
		[0.386,0.242], [0.381,0.216], [0.356,0.217], [0.376,0.195], [0.339,0.186], [0.349,0.168], [0.326,0.155], [0.270,0.146],
		[0.284,0.139], [0.310,0.145], [0.341,0.148], [0.328,0.128], [0.287,0.119], [0.310,0.109], [0.316,0.092], [0.348,0.124],
		[0.339,0.079], [0.352,0.102], [0.359,0.144], [0.379,0.177], [0.361,0.060], [0.375,0.085], [0.367,0.102], [0.381,0.068],
		[0.399,0.070], [0.398,0.100], [0.389,0.126], [0.379,0.149], [0.392,0.159], [0.409,0.134], [0.414,0.105], [0.426,0.128],
		[0.432,0.104], [0.426,0.092], [0.416,0.070], [0.430,0.051], [0.451,0.061], [0.465,0.086], [0.471,0.080], [0.486,0.066],
		[0.497,0.051], [0.495,0.081], [0.466,0.105], [0.466,0.119], [0.478,0.114], [0.441,0.143], [0.435,0.163], [0.392,0.179],
		[0.415,0.199], [0.517,0.086], [0.544,0.066], [0.555,0.087], [0.528,0.101], [0.520,0.118], [0.494,0.126], [0.510,0.141],
		[0.467,0.144], [0.460,0.150], [0.441,0.176], [0.466,0.182], [0.446,0.194], [0.429,0.215], [0.410,0.217], [0.401,0.240],
		[0.431,0.240], [0.445,0.233], [0.470,0.202], [0.485,0.190], [0.501,0.174], [0.501,0.159], [0.541,0.121], [0.576,0.084],
		[0.579,0.110], [0.554,0.130], [0.588,0.126], [0.575,0.144], [0.546,0.155], [0.525,0.179], [0.500,0.205], [0.471,0.226],
		[0.465,0.250], [0.434,0.271], [0.463,0.264], [0.439,0.283], [0.458,0.300], [0.475,0.291], [0.486,0.264], [0.503,0.235],
		[0.501,0.255], [0.480,0.306], [0.515,0.279], [0.532,0.291], [0.546,0.267], [0.524,0.255], [0.541,0.236], [0.519,0.219],
		[0.528,0.200], [0.544,0.205], [0.550,0.169], [0.566,0.166], [0.575,0.184], [0.555,0.217], [0.559,0.230], [0.560,0.251],
		[0.583,0.246], [0.591,0.227], [0.606,0.206], [0.588,0.196], [0.599,0.173], [0.590,0.143], [0.620,0.160], [0.826,0.096],
		[0.797,0.110], [0.760,0.141], [0.750,0.168], [0.738,0.204], [0.771,0.184], [0.794,0.149], [0.820,0.150], [0.820,0.120],
		[0.841,0.115], [0.840,0.136], [0.872,0.116], [0.865,0.101], [0.907,0.130], [0.899,0.151], [0.843,0.154], [0.843,0.176],
		[0.807,0.191], [0.799,0.204], [0.759,0.221], [0.767,0.229], [0.761,0.249], [0.795,0.236], [0.839,0.215], [0.868,0.200],
		[0.875,0.170], [0.903,0.188], [0.900,0.205], [0.914,0.216], [0.887,0.227], [0.881,0.235], [0.838,0.233], [0.812,0.241],
		[0.832,0.258], [0.765,0.263], [0.772,0.286], [0.760,0.301], [0.799,0.284], [0.824,0.279], [0.806,0.306], [0.782,0.318],
		[0.819,0.328], [0.828,0.319], [0.857,0.294], [0.866,0.275], [0.891,0.246], [0.889,0.269], [0.880,0.294], [0.874,0.319],
		[0.851,0.326], [0.830,0.354], [0.825,0.379], [0.815,0.404], [0.816,0.426], [0.794,0.430], [0.775,0.438], [0.761,0.451],
		[0.764,0.470], [0.756,0.485], [0.729,0.514], [0.767,0.501], [0.789,0.491], [0.801,0.465], [0.815,0.459], [0.836,0.436],
		[0.856,0.434], [0.846,0.460], [0.831,0.474], [0.807,0.496], [0.794,0.506], [0.802,0.529], [0.766,0.532], [0.738,0.540],
		[0.776,0.550], [0.774,0.566], [0.740,0.579], [0.730,0.596], [0.774,0.604], [0.799,0.579], [0.815,0.564], [0.841,0.547],
		[0.828,0.519], [0.846,0.500], [0.871,0.471], [0.885,0.484], [0.876,0.500], [0.909,0.515], [0.861,0.516], [0.875,0.526],
		[0.876,0.537], [0.855,0.544], [0.893,0.557], [0.896,0.581], [0.886,0.603], [0.855,0.573], [0.831,0.584], [0.840,0.597],
		[0.856,0.603], [0.879,0.633], [0.843,0.621], [0.812,0.617], [0.789,0.630], [0.756,0.608], [0.740,0.637], [0.762,0.637],
		[0.781,0.651], [0.769,0.672], [0.781,0.696], [0.818,0.705], [0.810,0.677], [0.809,0.649], [0.846,0.651], [0.841,0.675],
		[0.855,0.699], [0.881,0.650], [0.860,0.640], [0.504,0.522], [0.478,0.546], [0.469,0.574], [0.489,0.556], [0.512,0.546],
		[0.542,0.547], [0.525,0.565], [0.512,0.571], [0.480,0.593], [0.516,0.596], [0.492,0.615], [0.554,0.628], [0.542,0.595],
		[0.564,0.588], [0.583,0.611], [0.595,0.631], [0.573,0.580], [0.555,0.560], [0.571,0.547], [0.579,0.565], [0.617,0.555],
		[0.604,0.583], [0.130,0.054], [0.169,0.057], [0.135,0.068], [0.121,0.084], [0.143,0.086], [0.155,0.080], [0.180,0.069],
		[0.169,0.089], [0.145,0.100], [0.128,0.109], [0.136,0.121], [0.164,0.126], [0.155,0.106], [0.196,0.113], [0.181,0.089],
		[0.201,0.069], [0.202,0.089]
    ];
    
    let agg_y = [
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    	3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    	3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    	3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5,
    	5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    	5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
    ];

    let data = new Dataset();
    data.set(agg_x, agg_y);
    return data;
}

function get_iris() {
	let iris_x = [
		[ 5.100, 3.500, 1.400, 0.200], [ 4.900, 3.000, 1.400, 0.200], [ 4.700, 3.200, 1.300, 0.200], [ 4.600, 3.100, 1.500, 0.200],
		[ 5.000, 3.600, 1.400, 0.200], [ 5.400, 3.900, 1.700, 0.400], [ 4.600, 3.400, 1.400, 0.300], [ 5.000, 3.400, 1.500, 0.200],
		[ 4.400, 2.900, 1.400, 0.200], [ 4.900, 3.100, 1.500, 0.100], [ 5.400, 3.700, 1.500, 0.200], [ 4.800, 3.400, 1.600, 0.200],
		[ 4.800, 3.000, 1.400, 0.100], [ 4.300, 3.000, 1.100, 0.100], [ 5.800, 4.000, 1.200, 0.200], [ 5.700, 4.400, 1.500, 0.400],
		[ 5.400, 3.900, 1.300, 0.400], [ 5.100, 3.500, 1.400, 0.300], [ 5.700, 3.800, 1.700, 0.300], [ 5.100, 3.800, 1.500, 0.300],
		[ 5.400, 3.400, 1.700, 0.200], [ 5.100, 3.700, 1.500, 0.400], [ 4.600, 3.600, 1.000, 0.200], [ 5.100, 3.300, 1.700, 0.500],
		[ 4.800, 3.400, 1.900, 0.200], [ 5.000, 3.000, 1.600, 0.200], [ 5.000, 3.400, 1.600, 0.400], [ 5.200, 3.500, 1.500, 0.200],
		[ 5.200, 3.400, 1.400, 0.200], [ 4.700, 3.200, 1.600, 0.200], [ 4.800, 3.100, 1.600, 0.200], [ 5.400, 3.400, 1.500, 0.400],
		[ 5.200, 4.100, 1.500, 0.100], [ 5.500, 4.200, 1.400, 0.200], [ 4.900, 3.100, 1.500, 0.100], [ 5.000, 3.200, 1.200, 0.200],
		[ 5.500, 3.500, 1.300, 0.200], [ 4.900, 3.100, 1.500, 0.100], [ 4.400, 3.000, 1.300, 0.200], [ 5.100, 3.400, 1.500, 0.200],
		[ 5.000, 3.500, 1.300, 0.300], [ 4.500, 2.300, 1.300, 0.300], [ 4.400, 3.200, 1.300, 0.200], [ 5.000, 3.500, 1.600, 0.600],
		[ 5.100, 3.800, 1.900, 0.400], [ 4.800, 3.000, 1.400, 0.300], [ 5.100, 3.800, 1.600, 0.200], [ 4.600, 3.200, 1.400, 0.200],
		[ 5.300, 3.700, 1.500, 0.200], [ 5.000, 3.300, 1.400, 0.200], [ 7.000, 3.200, 4.700, 1.400], [ 6.400, 3.200, 4.500, 1.500],
		[ 6.900, 3.100, 4.900, 1.500], [ 5.500, 2.300, 4.000, 1.300], [ 6.500, 2.800, 4.600, 1.500], [ 5.700, 2.800, 4.500, 1.300],
		[ 6.300, 3.300, 4.700, 1.600], [ 4.900, 2.400, 3.300, 1.000], [ 6.600, 2.900, 4.600, 1.300], [ 5.200, 2.700, 3.900, 1.400],
		[ 5.000, 2.000, 3.500, 1.000], [ 5.900, 3.000, 4.200, 1.500], [ 6.000, 2.200, 4.000, 1.000], [ 6.100, 2.900, 4.700, 1.400],
		[ 5.600, 2.900, 3.600, 1.300], [ 6.700, 3.100, 4.400, 1.400], [ 5.600, 3.000, 4.500, 1.500], [ 5.800, 2.700, 4.100, 1.000],
		[ 6.200, 2.200, 4.500, 1.500], [ 5.600, 2.500, 3.900, 1.100], [ 5.900, 3.200, 4.800, 1.800], [ 6.100, 2.800, 4.000, 1.300],
		[ 6.300, 2.500, 4.900, 1.500], [ 6.100, 2.800, 4.700, 1.200], [ 6.400, 2.900, 4.300, 1.300], [ 6.600, 3.000, 4.400, 1.400],
		[ 6.800, 2.800, 4.800, 1.400], [ 6.700, 3.000, 5.000, 1.700], [ 6.000, 2.900, 4.500, 1.500], [ 5.700, 2.600, 3.500, 1.000],
		[ 5.500, 2.400, 3.800, 1.100], [ 5.500, 2.400, 3.700, 1.000], [ 5.800, 2.700, 3.900, 1.200], [ 6.000, 2.700, 5.100, 1.600],
		[ 5.400, 3.000, 4.500, 1.500], [ 6.000, 3.400, 4.500, 1.600], [ 6.700, 3.100, 4.700, 1.500], [ 6.300, 2.300, 4.400, 1.300],
		[ 5.600, 3.000, 4.100, 1.300], [ 5.500, 2.500, 4.000, 1.300], [ 5.500, 2.600, 4.400, 1.200], [ 6.100, 3.000, 4.600, 1.400],
		[ 5.800, 2.600, 4.000, 1.200], [ 5.000, 2.300, 3.300, 1.000], [ 5.600, 2.700, 4.200, 1.300], [ 5.700, 3.000, 4.200, 1.200],
		[ 5.700, 2.900, 4.200, 1.300], [ 6.200, 2.900, 4.300, 1.300], [ 5.100, 2.500, 3.000, 1.100], [ 5.700, 2.800, 4.100, 1.300],
		[ 6.300, 3.300, 6.000, 2.500], [ 5.800, 2.700, 5.100, 1.900], [ 7.100, 3.000, 5.900, 2.100], [ 6.300, 2.900, 5.600, 1.800],
		[ 6.500, 3.000, 5.800, 2.200], [ 7.600, 3.000, 6.600, 2.100], [ 4.900, 2.500, 4.500, 1.700], [ 7.300, 2.900, 6.300, 1.800],
		[ 6.700, 2.500, 5.800, 1.800], [ 7.200, 3.600, 6.100, 2.500], [ 6.500, 3.200, 5.100, 2.000], [ 6.400, 2.700, 5.300, 1.900],
		[ 6.800, 3.000, 5.500, 2.100], [ 5.700, 2.500, 5.000, 2.000], [ 5.800, 2.800, 5.100, 2.400], [ 6.400, 3.200, 5.300, 2.300],
		[ 6.500, 3.000, 5.500, 1.800], [ 7.700, 3.800, 6.700, 2.200], [ 7.700, 2.600, 6.900, 2.300], [ 6.000, 2.200, 5.000, 1.500],
		[ 6.900, 3.200, 5.700, 2.300], [ 5.600, 2.800, 4.900, 2.000], [ 7.700, 2.800, 6.700, 2.000], [ 6.300, 2.700, 4.900, 1.800],
		[ 6.700, 3.300, 5.700, 2.100], [ 7.200, 3.200, 6.000, 1.800], [ 6.200, 2.800, 4.800, 1.800], [ 6.100, 3.000, 4.900, 1.800],
		[ 6.400, 2.800, 5.600, 2.100], [ 7.200, 3.000, 5.800, 1.600], [ 7.400, 2.800, 6.100, 1.900], [ 7.900, 3.800, 6.400, 2.000],
		[ 6.400, 2.800, 5.600, 2.200], [ 6.300, 2.800, 5.100, 1.500], [ 6.100, 2.600, 5.600, 1.400], [ 7.700, 3.000, 6.100, 2.300],
		[ 6.300, 3.400, 5.600, 2.400], [ 6.400, 3.100, 5.500, 1.800], [ 6.000, 3.000, 4.800, 1.800], [ 6.900, 3.100, 5.400, 2.100],
		[ 6.700, 3.100, 5.600, 2.400], [ 6.900, 3.100, 5.100, 2.300], [ 5.800, 2.700, 5.100, 1.900], [ 6.800, 3.200, 5.900, 2.300],
		[ 6.700, 3.300, 5.700, 2.500], [ 6.700, 3.000, 5.200, 2.300], [ 6.300, 2.500, 5.000, 1.900], [ 6.500, 3.000, 5.200, 2.000],
		[ 6.200, 3.400, 5.400, 2.300], [ 5.900, 3.000, 5.100, 1.800]
	];
	let iris_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2];

	for (let i = 0; i < iris_y.length; i++) {
		if (iris_y[i] == 0) {
			iris_y[i] = "Iris-setosa";
		}
		else if (iris_y[i] == 1) {
			iris_y[i] = "Iris-versicolor";
		}
		else {
			iris_y[i] = "Iris-virginica";
		}
	}

	let data = new Dataset();
    data.set(iris_x, iris_y);
    return data;	
}

/**
    Returns the dataset with the specified name.
*/
function get_dataset(name) {
    if (name == "spiral") {
        return get_spiral();
    }
    else if (name == "flame") {
        return get_flame();
    }
    else if (name == "moons") {
        return get_moons();
    }
    else if (name == "gaussian") {
        return get_gaussian();
    }
    else if (name == "aggregation") {
        return get_aggregation();
    }
    else if (name == "circle") {
        return get_circle();
    }
    else {
        throw("Unknown dataset: " + name);
    }
}

/**
    Returns the render options for the specified dataset.
*/
function get_render_options(name) {
    if (name == "spiral") {
        return [2.2, 2.2, -1.1, -1.1];
    }
    else if (name == "flame") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "moons") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "circle") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "gaussian") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "aggregation") {
        return [1.0, 1.0, 0, -0.1];
    }
    else {
        throw("Unknown dataset: " + name);
    }
}

/**
	Returns the default hyperparameter settings for classifier and datasets combinations.
*/
function get_settings(type, name) {
    if (type == "knn") {
        return [3];
    }
    else if (type == "linear") {
        if (name == "spiral") return [50, 0.8];
        if (name == "gaussian") return [500, 0.8];
        if (name == "aggregation") return [4000, 1.0];
        if (name == "circle") return [40, 0.8];
        if (name == "moons") return [800, 0.7];
        return [500, 0.8];
    }
    else if (type == "nn") {
        if (name == "spiral") return [72, 1600, 0.8];
        if (name == "moons") return ["32,16", 1500, 0.3];
        if (name == "gaussian") return [64, 800, 0.8];
        if (name == "flame") return ["16,8", 600, 0.3];
        if (name == "circle") return ["16,16", 400, 0.3];
        if (name == "aggregation") return [16, 400, 0.8];
        return [32, 500, 0.8];
    }
    else if (type == "dt") {
    	if (name == "flame") return [5, 5];
    	if (name == "circle") return [5, 5];
        return [7, 5];
    }
    else if (type == "rf") {
    	if (name == "spiral") return [7, 9, 5];
    	if (name == "flame") return [11, 7, 5];
    	if (name == "moons") return [17, 7, 5];
    	if (name == "gaussian") return [19, 9, 5];
    	if (name == "circle") return [7, 9, 5];
    	if (name == "aggregation") return [7, 7, 5];
        return [11, 7, 5];
    }
    else if (type == "svm") {
        if (name == "flame") return [1000];
        if (name == "moons") return [200];
        if (name == "gaussian") return [20];
        if (name == "circle") return [20];
        if (name == "aggregation") return [80];
        return [40];
    }
}

/**
	Returns the batch size for iterable classifiers.
*/
function get_iteration_stepsize(type, name) {
	if (type == "linear") {
        if (name == "aggregation") return 50;
        if (name == "flame") return 5;
        if (name == "moons") return 10;
        if (name == "gaussian") return 10;
        return 1;
    }
    else if (type == "nn") {
    	if (name == "gaussian") return 20;
    	if (name == "circle") return 5;
        return 20;
    }
}

/** ------------------------------------------------------

Utility functions and classes for classifiers.

--------------------------------------------------------- */

/**
    Holds a loaded and converted dataset.
*/
class Dataset {
    // Constructor
    constructor() {
        this.x = [];
        this.y = [];
        this.labels = new Labels();
    }

    // Sets input and labels data to a dataset
    set(nx, ny) {
    	for (let i = 0; i < ny.length; i++) {
    		this.add_example(nx[i], ny[i]);
    	}
    }

    // Returns a copy of this dataset
    clone() {
    	let nx = [];
    	let ny = [];

    	// Copy all instances
    	for (let i = 0; i < this.no_examples(); i++) {
    		// Copy attributes of current instance
    		let e = this.x[i];
    		let new_e = [];
    		for (let a = 0; a < e.length; a++) {
    			new_e.push(e[a]);
    		}

    		// Add to new data arrays
    		nx.push(new_e);
    		ny.push(this.y[i]);
    	}

    	// Create new dataset
    	let new_data = new Dataset();
        new_data.x = nx;
        new_data.y = ny;
        new_data.labels = this.labels;

        return new_data;
    }

    // Randomly shuffles the dataset examples
    shuffle() {
        // Create new arrays
        let nx = [];
        let ny = [];
        // Holds which instances that have been copied or not
        let done = new Array(this.y.length).fill(0);
        
        // Continue until all ínstances have been copied
        while (nx.length < this.x.length) {
            // Find a random instance that has not been copied
            let i = -1;
            while (i == -1) {
                let ti = Math.floor(rnd() * this.x.length);
                if (done[ti] == 0) {
                    // Not copied. Use this index.
                    done[ti] = 1;
                    i = ti;
                }
                else {
                    // Already copied. Get new index.
                    i = -1;
                }
            }
            
            // Get values
            let xv = this.x[i];
            let yv = this.y[i];
            
            // Copy to new arrays
            nx.push(xv);
            ny.push(yv);
        }

        this.x = nx;
        this.y = ny;
    }

    // Splits the dataset into a training and test dataset.
    train_test_split(ratio) {
        let no_test = Math.floor(this.no_examples() * ratio);
        
        let x_test = [];
        let y_test = [];
        let x_train = [];
        let y_train = [];

        for (let i = 0; i < this.no_examples(); i++) {
            if (i < no_test) {
                // Copy to test set
                x_test.push(this.x[i]);
                y_test.push(this.y[i]);
            }
            else {
                // Copy to training set
                x_train.push(this.x[i]);
                y_train.push(this.y[i]);
            }
        }

        // Create training data
        let training_data = new Dataset();
        training_data.x = x_train;
        training_data.y = y_train;
        training_data.labels = this.labels;

        // Create test data
        let test_data = new Dataset();
        test_data.x = x_test;
        test_data.y = y_test;
        test_data.labels = this.labels;

        return [training_data, test_data];
    }

    // Returns a training and test dataset for a specified fold in cross-validation.
    get_fold(fold_no, folds) {
        let no = Math.floor(this.no_examples() / folds);

        // Get fold start and end
        let i_start = no * (fold_no - 1);
        let i_end = i_start + no;
        if (fold_no == folds) {
        	i_end = this.no_examples();
        }

        let x_test = [];
        let y_test = [];
        let x_train = [];
        let y_train = [];

        for (let i = 0; i < this.no_examples(); i++) {
            if (i >= i_start && i < i_end) {
                // Copy to test set
                x_test.push(this.x[i]);
                y_test.push(this.y[i]);
            }
            else {
                // Copy to training set
                x_train.push(this.x[i]);
                y_train.push(this.y[i]);
            }
        }

        // Create training data
        let training_data = new Dataset();
        training_data.x = x_train;
        training_data.y = y_train;
        training_data.labels = this.labels;

        // Create test data
        let test_data = new Dataset();
        test_data.x = x_test;
        test_data.y = y_test;
        test_data.labels = this.labels;

        return [training_data, test_data];
    }

    // Creates a subset of this dataset.
    create_subset(nx, ny) {
    	let d = new Dataset();
    	d.x = nx;
    	d.y = ny;
    	d.labels = this.labels;
    	return d;
    }

    // Adds an example (input and label) to this dataset.
    add_example(ex_x, ex_y) {
        this.x.push(ex_x);
        this.y.push(this.label_to_id(ex_y));
    }

    // Returns number of unique classes in this dataset.
    no_classes() {
        return this.labels.dist.length;
    }

    // Returns thenumber of examples (size) of this dataset.
    no_examples() {
        return this.x.length;
    }

    // Returns the number of attributes in this dataset.
    no_attr() {
        return this.x[0].length;
    }

    // Returns the class label for a class id.
    id_to_label(id) {
        for (let i = 0; i < this.labels.dist.length; i++) {
            let le = this.labels.dist[i];
            if (le.id == id) {
                return le.label;
            }
        }
        console.log("Should not happen...");
        return id;
    }

    // Returns the class id for a class label.
    label_to_id(label) {
        for (let i = 0; i < this.labels.dist.length; i++) {
            let le = this.labels.dist[i];
            if (le.label == label) {
                return le.id;
            }
        }

        // Add new
        let id = this.labels.size();
        this.labels.dist.push( {label: label, id: id, cnt: 0} );
        return id;
    }

    // Feature-wise normalization where we subtract the mean and divide with stdev
    normalize() {
        for (let c = 0; c < this.no_attr(); c++) {
            let m = this.attr_mean(c); // mean
            let std = this.attr_stdev(c, m); // stdev
            for (let i = 0; i < this.no_examples(); i++) {
                this.x[i][c] = (this.x[i][c] - m) / std;
            }
        }
    }

    // Calculates the mean value of an attribute
    attr_mean(c) {
        let m = 0;
        for (let i = 0; i < this.no_examples(); i++) {
            m += this.x[i][c];
        }
        m /= this.no_examples();
        return m;
    }

    // Calculates the standard deviation of an attribute
    attr_stdev(c, m) {
        let std = 0;
        for (let i = 0; i < this.no_examples(); i++) {
            std += Math.pow(this.x[i][c]  - m, 2);
        }
        std = Math.sqrt(std / (this.no_examples() - 1));
        return std;
    }
}

/**
    Holds a dataset instance.
*/
class Instance {
    constructor(x, label) {
        this.x = x;
        this.label = label;
        this.dist = 0; //Used by KNN
    }
}

/**
    Holds the unique classes (labels).
*/
class Labels {
    constructor() {
        this.dist = [];
    }
    
    // Returns the number of unique classes
    size() {
        return this.dist.length;
    }
    
    // Resets the class count (used for predictions)
    reset_cnt() {
        for (let i = 0; i < this.dist.length; i++) {
            this.dist[i].no = 0;
        }
    }
    
    // Increases the label count (used for predictions)
    inc_cnt(id) {
        for (let i = 0; i < this.dist.length; i++) {
            if (this.dist[i].id == id) {
                this.dist[i].no += 1;
            }
        }
    }
    
    // Returns the label with highest count (used for predictions)
    get_best() {
        let max = -1;
        let index = -1;
        
        for (let i = 0; i < this.dist.length; i++) {
            if (this.dist[i].no > max) {
                max = this.dist[i].no;
                index = i;
            }
        }

        // To be sure some class is returned
        if (index < 0) {
            return 0;
        }

        return this.dist[index].id;
    }
}

/**
    Returns a radnom batch of training instances from the dataset.
    Used for batch training by the Linear and Neural Network classifiers.
*/
function get_batch(x1, y1, batch_size) {
	// Batch size is equal to or larger than
	// dataset size -> return dataset
	if (batch_size >= x1.columns()) {
		return [x1, y1];
	}

	// Create new arrays
    let nx = NN_vec2D.zeros(x1.rows(), batch_size);
    let ny = NN_vec1D.zeros(batch_size);

    // Copy random examples
    let cnt_i = 0;
    for (let n = 0; n < batch_size; n++) {
    	let ci = Math.floor(rnd() * x1.columns());
    	// Copy attributes
    	for (let r = 0; r < x1.rows(); r++) {
    		nx.set(r, cnt_i, x1.get(r, ci));
    	}
        ny.set(cnt_i, y1.get(ci));
        cnt_i++;
    }

    return [nx, ny];
}

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

            e.prob = p;
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

/** ------------------------------------------------------

Functions for rendering dataset and decision boundaries.

--------------------------------------------------------- */

// Init variables
var cellw = 5;
var canvas;
var ctx;

/**
    Clears the drawing canvas.
*/
function clear() {
    document.getElementById("acc").innerHTML = "&nbsp;";
    document.getElementById("citer").innerHTML = "&nbsp;";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
    Inits the drawing canvas.
*/
function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
}

/**
    Draws the decision boundaries map.
*/
function draw_map(map) {
    for (let i = 0; i < map.length; i++) {
        let x1 = parseInt(i % 100);
        let x2 = parseInt(i / 100);
        drawcell(x1, x2, map[i]);
    }
}

/*
    Draws a filled cell with a different color for each class value.
*/
function drawcell(x, y, val) {
    let x1 = x * cellw;
    let y1 = y * cellw;
    
    // Select color based on class value
    let color = "#FFFFFF";
    if (val == 0) color = "#FFDBC2"; // red
    if (val == 1) color = "#C2DBFF"; // blue
    if (val == 2) color = "#DBFFC2"; // green
    if (val == 3) color = "#FAFC83"; // yellow
    if (val == 4) color = "#F7D4F3"; // pink
    if (val == 5) color = "#CCCCCC"; // gray
    if (val == 6) color = "#C3F4F3"; // cyan
    
    // Draw square
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1, cellw, cellw);
    ctx.closePath();
}

/**
    Draws the labels from the dataset.
*/
function draw_labels(data, opt) {
    for (let i = 0; i < data.no_examples(); i++) {
        let xe = data.x[i];
        let ye = data.y[i];
        let x1 = (xe[0] - opt[2]) / opt[0] * 100;
        let x2 = (xe[1] - opt[3]) / opt[1] * 100;
        drawlabel(x1, x2, ye);
    }
}

/**
    Draws a single label from the dataset.
*/
function drawlabel(x, y, val) {
    let x1 = x * cellw;
    let y1 = y * cellw;
    
    // Draw border
    let r = cellw / 2;
    let c = cellw / 2;
    ctx.beginPath();
    ctx.fillStyle = "#000000";
    ctx.arc(x1 + c, y1 + c, r, 0, 2 * Math.PI, false);
    ctx.fill();
    ctx.closePath();
    
    // Select color based on actual class
    let fcolor = "#FFFFFF";
    if (val == 0) fcolor = "#CC6600"; // red
    if (val == 1) fcolor = "#0066CC"; // blue
    if (val == 2) fcolor = "#66CC00"; // green
    if (val == 3) fcolor = "#ABAD1F"; // yellow
    if (val == 4) fcolor = "#C45EB8"; // pink
    if (val == 5) fcolor = "#888888"; // gray
    if (val == 6) fcolor = "#2CD6D0"; // cyan
    
    // Draw filled circle
    r = cellw / 2 - 1;
    ctx.beginPath();
    ctx.fillStyle = fcolor;
    ctx.arc(x1 + c, y1 + c, r, 0, 2 * Math.PI, false);
    ctx.fill();
    ctx.closePath();
}
