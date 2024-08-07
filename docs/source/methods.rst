Methods
=====

.. autoclass:: lumache
   :members:

   {% block functions %}
   {% if functions %}

   Functions
   ---------

   {% for item in functions %}

   .. autofunction:: {{ item }}


   .. include:: backreferences/{{fullname}}.{{item}}.examples

   .. raw:: html

               <div class="sphx-glr-clear"></div>

   {%- endfor %}
   {% endif %}
   {% endblock %}
