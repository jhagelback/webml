
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
