
### Useful Invocations to Remember
#### Regenerate requirements.txt
`python -m  pipreqs.pipreqs . --force --ignore .venv --mode compat`
Note: numpy~=1.26.4

#### Time Trials
`pytest time_trials [--benchmark_histogram]`

#### Cython Compile
`cythonize -i [filename.py]`

#### Unit Tests
run: `pytest --benchmark-skip`
debug: `py.test --pdb`


#### design considerations
##### information flow
a node should be able to take multiple inputs at the same time. In practice, what does this mean?
One way that a node can take multiple inputs - in fact, the traditional way - is that each input
connection "feeds" to the same place, they are all summed (and perhaps normalized or scaled) to
produce what is essentially just one input. In such a case, any number of inputs should be
acceptable.


Another way that a node might take multiple inputs is in parallel, each input considered on its
own. In such a case it makes sense to keep the inputs separate and distinct.


Because both of these behaviors is reasonable, the medium of communication between nodes is most
easily organized into a dict where the entries are parallel channels, keys being the arbitrary
channel names (though I'll use ints for simplicity) and the values are unbounded lists of ndarrays.
Nodes should therefore always accept these dicts and be responsible for selecting what to use;
functions used by pipelines, however, should have strict requirements and rely on the pipeline to 
manage their input/output.

Does this design require "synapse" functions, whose job it is to convert the input dict into a
specific input format expected by a pipeline function? In this view, I think it makes sense for
synapse functions to have a special place in pipeline nodes, but at the same time I see an 
argument for making them more general, using them as general pipeline functions as I have been so
far. This idea *feels* good, but I think it falls short in that intra-pipeline synapse functions
would be operating not on the typical dict of lists, but on a single entry (list or ndarray).
While this difference does not feel like enough of a reason to differentiate them (particularly
in a pythonic context) I think the compartmentalization will likely make some considerations
easier in the future - backpropagation and optimization in particular. 

In this paradigm, how should updates be handled? Some pipeline functions may require a kind of
non-standard update, so should updates be a dict created by the pipeline function that labels
the updates unambiguously? I think this makes sense.