## Web ML Demonstrator

### About
Web ML Demonstrator is a machine learning demonstrator running purely on the client browser. All algorithms are implemented in JavaScript for the purpose of this demonstrator. They are not optimized for high performance and don't have all the functionality of state-of-the-art implementations. The main purpose of this demonstrator is to be used as a tool when teaching and explaining machine learning and machine learning related concepts. 

### Testing it

[Experimenter](http://aiguy.org/webml/experimenter.html): A web application where you can upload datasets in csv format and run machine learning experiments on them.

[Visualizer](http://aiguy.org/webml/index.html): A web application where you can see visualizations of how different machine learning algorithms learn on a number of two-dimensional datasets.

### Running it locally
Web ML Demonstrator runs entirely in the browser and doesn't need a web server. Just open the html files in a browser to run it locally.

### Code modifications
All modifications to the code shall be done in the JavaScript files in the js folder. When the code modifications are finished, the jscombiner.py Python script must be run to combine all js files into one. 

The library version can be changed in the Python script. Note that if you change version, you must also change the filename of the JavaScript library in the html files.
