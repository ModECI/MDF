{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :special-members: __call__, __add__, __mul__

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {%- if not item.startswith('_') %}
        {%- if item not in inherited_members %}
            ~{{ name }}.{{ item }}
        {%- endif -%}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
     {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
     {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
