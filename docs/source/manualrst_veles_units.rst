Units
:::::

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
