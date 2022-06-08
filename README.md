# Homogenization Methods for Heat Lens and Optimal Transportation

A python package for resolving variable coefficients for the diffusion equation.

# Install

Current prepackage install:

```bash
pip3 install -e .
```

# Use

After installation import `homogenize`

```bash
import homogenize
```
# Tests

To ensure that all methods run properly, consider runing the following with `python` or `python3` depending on your python executable.

```bash
sh ./test.sh python3
```

Individual tests can be found in `./bin/`

# Examples

To generate a complete set of examples, consider runing the following with `python` or `python3` depending on your python executable.

```bash
python3 ./examples/discrete_laplacian.py
```

Individual examples can be found in `./bin/`

See `./doc/` for a presentation of the listed examples.

# Directory Structure

```bash
──homogenize
  │   design.py
  │   heatlens.py
  │   misc.py
  │   pdeint.py
  │   transport.py
  └───
```

`design.py` : Handles all adaptive step methods

`heatlens.py` : Implements `odeint_adjoint` from `torchdiffeq` package (can be reduced through modification)

`misc.py` : Implements `BDF5Solver` the adaptive step bdf method.

`pdeint.py` : Implements `DOPRI5Solver` the adaptive step Dormant Prince method from `torchdiffeq`

`transport.py` : Implements several miscellaneous methods including norms, verification methods and the like.

# Up Coming

Licensing and requirements to be updated.
