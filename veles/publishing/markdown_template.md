{% set set_names = ("test", "validation", "train") %}

Veles workflow report - {{ name }}
========================{{ "=" * name|length }}

Task
----

{% if image is not none %}
<img style="float: left;" src="{{ image["name"] }}" alt="Task image" id="task_image">
{% endif %}

!!! block ""
    #### Description
    
    {{ description }}

!!! block ""
    #### Workflow path
    
    {{ workflow_file }}

!!! block ""
    #### Configuration path
    
    {{ config_file }}

{% macro ellipsis(value, maxlen) %}{{ value
if value|string|length < maxlen else
"%s..." % (value|string)[:maxlen - 3]}}{% endmacro %}
{% macro td(value, maxlen) %}{{
"%s%s" % (ellipsis(value, maxlen), " " * max(0, maxlen - value|string|length))
}}{% endmacro %}
{% macro tdc(value, maxlen) %}{{
"%s%s%s" % (" " * max(0, ((maxlen - value|string|length) / 2)|round|int),
            ellipsis(value, maxlen),
            " " * max(0, (maxlen - value|string|length - ((maxlen - value|string|length) / 2)|round))|round|int)
}}{% endmacro %}
{% macro tdr(value, maxlen) %}{{
"%s%s" % (" " * max(0, maxlen - value|string|length), ellipsis(value, maxlen))
}}{% endmacro %}
!!! block ""
    Results
    -------
    
    |        metric        |     set     |             value             |
    |:---------------------|:------------|:------------------------------|
    {% for key, set_vals in results | dictsort %}
        {% if set_vals is mapping %}
            {% for set_name, val in set_vals | dictsort %}
    | {{ td(key, 20) }} | {{ td(set_name, 11) }} | {{ td(val, 29) }} |
            {% endfor %}
        {% else %}
    | {{ td(key, 20) }} |      -      | {{ td(set_vals, 29) }} |
        {% endif %}
    {% endfor %}

!!! block ""
    Source data
    -----------
    
    #### Samples
    
    |      |{% for name in set_names %} {{ tdc(name, 11) }} |{% endfor %}    total    |
    |:----:|:-----------:|:-----------:|:-----------:|:-----------:|
    | size |{% for i in range(3) %} {{ tdc(class_lengths[i], 11) }} |{% endfor %} {{ tdc(total_samples, 11) }} |

!!! block ""
    #### Labels
    
    Total: {{ labels|length }}

|        label        |{% for i in range(3) %}{% if label_stats[i]|length > 0 %} {{ tdc("size (%s)" % set_names[i], 18) }} |{% endif %}{% endfor %}

|--------------------:|{% for i in range(3) %}{% if label_stats[i]|length > 0 %}-------------------:|{% endif %}{% endfor %}

{% for label in labels|sort %}
| {{ tdr(label, 19) }} |{% for i in range(3) %}{% if label_stats[i]|length > 0 %} {{ tdr(label_stats[i][label], 18) }} |{% endif %}{% endfor %}

{% endfor %}

!!! block ""
    #### Normalization
    
    |  domain  |        type        |          parameters          |
    |:--------:|:------------------:|:----------------------------:|
    |  samples | {{ tdc(normalization, 18) }} | {{ tdc(normalization_parameters, 28) }} |
    {% if target_normalization is defined %}
    |  targets | {{ tdc(target_normalization, 18) }} | {{ tdc(target_normalization_parameters, 28) }} |
    {% endif %}

!!! block ""
    Run stats
    ---------
    
    #### Elapsed time
    
    {{ days }} days, {{ hours }} hours, {{ mins }} minutes, {{ secs|round(1) }} secs
    
    |            unit            |      time      |
    |----------------------------|:---------------|
    {% for name, value in unit_run_times_by_class|dictsort(by='value')|reverse %}
    | {{ td(name, 26) }} | {{ td(value|round(4), 14) }} |
    {% endfor %}

Configuration
-------------

```
:::python
{{ config_text }}
```

!!! block ""
    Random seeds
    ------------
    
    | index |          Value (binascii.hexlify)          |
    |------:|:-------------------------------------------|
    {% for value in seeds %}
    | {{ tdr(loop.index, 5) }} | {{ td(hexlify(value).decode('charmap'), 42) }} |
    {% endfor %}

!!! block ""
    Workflow scheme
    ---------------
    
    <img src="workflow.{{ imgformat }}" id="workflow_scheme">

Plots
-----

{% for name in plots|sort %}
!!! block ""
    #### {{ name }}
    
    <img src="{{ name }}.{{ imgformat }}" class="plot">

{% endfor %}
