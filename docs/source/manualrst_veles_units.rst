Units
:::::

Feed-forward units
******************

* All-to-All perceptron layers (:mod:`veles.znicz.all2all`).
* Activation functions (:mod:`veles.znicz.activation`).
* Convolutional layers (:mod:`veles.znicz.conv`).
* Pooling layers (:mod:`veles.znicz.pooling`).
* Evaluators (:mod:`veles.znicz.evaluator`), softmax and MSE are implemented.


Gradient descent units
**********************

This units calculate gradient descent via back-propagation of gradient signals.
For **each** feed-forward layer should be a coupled gradient descent unit.

* GD for perceptron layers (:mod:`veles.znicz.gd`).
* GD for activation functions (:class:`veles.znicz.activation.ActivationBackward`).
* GD for convolutional layers (:mod:`veles.znicz.gd_conv`).
* GD for pooling layers (:mod:`veles.znicz.gd_pooling`).


Service units
*************

* Start point and end point
	* Start point is runned once on `Workflow.run()`. To run your units, you should link first of them from the start point.
	* End point should be runned when your calculation is finished.
* `Loaders` load raw images, pre-process them and make the initial data vectors.
* `Repeater` is a dummy unit that should be linked from `start_point` and from the last unit of the Workflow.
* `Decision` decides whether to stop training or continue.
* `Snapshotter` makes `pickle` snapshots from the `Workflow` each epoch.
* `Plotters` are used to draw plots: weight matrices, error for epochs, etc.
