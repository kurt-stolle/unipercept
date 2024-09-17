# UniPercept

This package contains a collection of libraries for rapid testing and development in computer vision research settings. 

Note that this is explicitly not a 'framework', but rather a collection of common utlities and boilerplate code. 
The individual libraries can be individually used and combined, without requiring a large shift in paradigm or major cross-dependencies.

Additionally, `unipercept` includes Hydra-based configurataion system and CLI to speed up common research tasks.

## Installation

This package requires at least Python 3.11 and PyTorch 2.1. Once you have created an environment with these 
dependencies, we can proceed to install `unipercept` using one of three installation methods.

### Stable release (recommended)
You can install the latest stable release from PyPI via
```bash
pip install unipercept
```

### Master branch
To install the latest version, which is not guaranteed to be stable, install from GitHub using 
```bash
pip install git+https://github.com/kurt-stolle/unipercept.git
```

### Developers
If your use-case requires changes to our codebase, we recommend that you first fork this repository and download your
own fork locally. Assuming you have the GitHub CLI installed, you can clone your fork with
```bash
gh repo clone unipercept
```
Then, you can proceed to install the package in editable mode by running
```bash
pip install --editable unipercept
```
You are invited to share your improvements to the codebase through a pull request on this repository. 
Before making a pull request, please ensure your changes follow our code guidelines by running `pre-commit` before 
adding your files into a Git commit.

## Training and evaluation

Models can be trained and evalurated from the CLI or through the Python API. 

### CLI
To train a model with the CLI:
```bash
unipercept train --config <config path> <overrides>
```
Without a `<config name>`, an interactive prompt will be started to assist in finding a configuration file.

To evaluate a model with the CLI:
```bash
unipercept evaluate --config <config path> <overrides>
```

## Developer guidelines
All tests can ran via `python -m pytest`. 
However, we also provide a `make` directive that uses `pytorch-xdist` to speed up the process:
```
make test
```
You may need to tune the parameters if memory problems arise during testing. 

Similarly, benchmarks are implemented using `pytest-benchmark`. To run them, use:
```
make benchmark
```

Coverage reports are built using `pytest-cov`, and executed using:
```
make coverage
```

Last, we largely follow the same principles and methods as [Transformers](https://huggingface.co/docs/transformers) uses for testing. 
For more information on using `pytest` for automated testing, check out [their documentation](https://huggingface.co/transformers/v3.4.0/testing.html).

## F.A.Q.

###### Building fais in Conda environemnts that ship their own GCC/G++ versions

If you are using a Conda environment that ships its own GCC/G++ versions, you may encounter issues when building the
torch extensions shipped with `unipercept`. To resolve this, you can set the `CC` and `CXX` environment variables to the
active environment's GCC/G++ versions before running the installation command. 

In your Conda environment, run the following commands:

```bash
conda env config vars set CC=$(which gcc)
conda env config vars set CXX=$(which g++)
```

Then, reload the environment and install `unipercept` as usual.
## License

This repository is released under the MIT License. For more information, please refer to the LICENSE file.
