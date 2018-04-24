### References

Daniel Shiffman's implementation of Boids can be seen in your browser, and underlies my port to Numpy. (In particular, the default parameters are equivalent - go see if you believe my analysis in the live setting!)
https://processing.org/examples/flocking.html

SciPy's hierarchical clustering works in non-Euclidian space. I am indebted to Jörn Hees for his clear tutorial.
https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

The correct 1d periodic metric comes from this question, because it's always the little things.
https://math.stackexchange.com/questions/11073/unit-circle-metric

Ellipse-fitting by the PCA in numpy.linalg is a work in progress, adapting pichenettes' code.
https://dsp.stackexchange.com/a/2290

Some future directions:
* analysis of the implicit force field via numdifftools
    * https://stackoverflow.com/a/39558950
* port Sim to theano-like gpu-compiler, run on 5x5x5 parameter grid
    * summarize runs in terms of cluster eigenvalues at 'good timestep'