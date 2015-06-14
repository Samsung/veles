{% set set_names = ("test", "validation", "train") %}

Veles workflow report
=====================

Task
----

{% if image is not none %}
<img style="float: left;" src="{{ image["name"] }}" alt="Task image">
{% endif %}

#### {{ description }}

#### Workflow path

{{ workflow_file }}

#### Configuration path

{{ config_file }}

Results
-------
