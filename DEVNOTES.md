
### Useful Invocations to Remember
#### Regenerate requirements.txt
`python -m  pipreqs.pipreqs . --force --ignore .venv`
Note: numpy==1.26.4

#### Time Trials
`pytest time_trials [--benchmark_histogram]`

#### Cython Compile
`cythonize -i [filename.py]`

#### Unit Tests
run: `pytest --benchmark-skip`
debug: `py.test --pdb`
