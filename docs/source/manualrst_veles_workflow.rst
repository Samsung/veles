:orphan:

Workflow
::::::::

Each workflow has a start point and an end point (:class:`StartPoint <veles.workflow.StartPoint>`
and :class:`EndPoint <veles.workflow.EndPoint>`), so that when :meth:`run() <veles.workflow.Workflow.run>`
is called, the start point begins the party and the end point's run ends the party,
triggering :meth:`on_workflow_finished() <veles.workflow.Workflow.on_workflow_finished>`. 