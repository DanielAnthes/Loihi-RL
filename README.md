# Loihi-RL

> Iterating through new ideas. Seeing how they work out. Then start all over again. Get a spike of happiness if it works. 

Nengo implementation of reinforcement learning in watermaze tasks using mouses.
Intended to be run on the Loihi chip as well.

## Installation

This project is intended to be run using NxSDK 0.9, which requires Python 3.5.2.

The [nengo-loihi installation page](https://www.nengo.ai/nengo-loihi/v0.9.0/installation.html) suggests using miniconda, but unfortunately miniconda does not actually contain 3.5.2 in its repository (not for Arch at least).

So for me the following (using *mini* conda) does not resolve, because the package repositories do not contain the correct version:

`conda create --name loihi python=3.5.2`

So at this point we should resort to building Python 3.5.2 from source.
But since I don't care about maintaining a global Python 3.5.2 install, I instead checked out `pyenv` which does contain it:

`pyenv install --list | grep 3.5.2`

```
3.5.2
anaconda3-5.2.0
pypy3.3-5.2-alpha1-src
pypy3.3-5.2-alpha1
```

`pyenv` allows us to activate Python 3.5.2 only for this project and will do the compiling for us.
Install with:

`pyenv install -v 3.5.2`

N.B. my first build failed due to the correct version of OpenSLL not being found.
I had a look at [common build problems for pyenv](https://github.com/pyenv/pyenv/wiki/Common-build-problems) and followed instructions for my system (Arch).
I double checked I installed all dependencies and also made sure to point to the correct (older) version of OpenSLL.
Long ive downgrading.

```
LDFLAGS="-L/usr/lib/openssl-1.0" \
CFLAGS="-I/usr/include/openssl-1.0"\
pyenv install -v 3.5.2
```

Setting `LDFLAGS` and `CFLAGS` was necessary for successful compilation for me.

Activate as your global python version using: 

```
pyenv global 3.5.2
```

(Go back with to the default system python installation using: `pyenv global system`).
Or only use 3.5.2 locally (recommended; assumes you are in our NMC project repo):

```
pyenv local 3.5.2
```

This creates a `.python-version` file.

From here on you can install packages with your preferred method, either straight into the 3.5.2. installation with pip, or in some virtual environment.

To ensure `python` indeed refers to the one selected with `pyenv`, add the following to your `.bashrc` or `.zshrc`:

```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
