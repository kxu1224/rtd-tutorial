Methods
=====

.. automethod:: syng_bts.python.Experiments_new.PilotExperiment

.. _pilot:

PilotExperiment
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

.. _apply:

ApplyExperiment
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']


.. _transfer:

Transfer
----------------