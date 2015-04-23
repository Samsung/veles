{{ description }}

Achieved accuracy: {{ errors_pt }}%

Labels: {{ labels }}

Plots:

{% for name in plots %}
!{{ name }}.png|width=600,height=600!
{% endfor %}