
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

