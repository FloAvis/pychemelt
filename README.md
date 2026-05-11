# PyCheMelt

[![Tests](https://github.com/osvalB/pychemelt/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/osvalB/pychemelt/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/osvalB/pychemelt/graph/badge.svg)](https://codecov.io/gh/osvalB/pychemelt)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://osvalb.github.io/pychemelt)

Welcome to PyCheMelt, a python-package to globally analyse protein denaturation data.

## Features

- **Data Import**: Import differential scanning fluorimetry (DSF) data or circular dichroism (CD) data from various instruments, including Nanotemper (Prometheus and Panta), Unchained Labs (AUNTY and UNCLE), Applied Photophysics (SUPR-DSF and Chirascan), and Jasco instruments.   
- **Thermochemical denaturation of protein monomers**: Fit the melting curves obtained at different denaturant agent concentrations to a thermochemical denaturation model to determine the thermodynamic parameters of protein unfolding, such as the melting temperature (<em>T</em><sub>m</sub>), the change in enthalpy (Δ<em>H</em>), and the change in heat capacity (Δ<em>C</em><sub>p</sub>).
- **Thermal denaturation of oligomers**: Fit the melting curves obtained at different protein concentrations to reversible two-state or three-state unfolding models.

## Install for Users

You can install PyCheMelt using pip

```bash
pip install pychemelt
```
 Or using [uv](https://docs.astral.sh/uv/)

```bash
uv add pychemelt
```

## Examples

Jupyter notebooks with examples of how to use PyChemelt are available at https://github.com/osvalB/pychemelt_analyses


## Install from Source - Development

Run these commands in your shell:

```bash
git clone https://github.com/osvalB/pychemelt.git
cd pychemelt
uv sync --extra dev
```

Verify the installation:

```bash
uv run pytest -v
```

## Authors

- Osvaldo Burastero
- Florian Vögele 