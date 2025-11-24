# MIG Optimization using E-graphs

This project implements and compares different approaches for Majority-Inverter Graph (MIG) optimization, including traditional methods and novel e-graph-based techniques.

## Project Structure

```
project/
├── emig/               # C++ baseline implementation
├── mig/                # Python frontend
├── mig_egg/            # Rust e-graph engine
└── mignite/           # MIG infrastructure
```

## Requirements

* pygraphviz
* pydot
* circuitgraph


### System Dependencies
```bash
# Install CBC solver
conda install -c conda-forge coincbc coin-or-cbc

# Install system dependencies
sudo apt install coinor-libcbc-dev
sudo apt install zlib1g-dev yosys

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Python Dependencies
```bash
# Install graphviz (for Mac users)
export GRAPHVIZ_DIR="/opt/homebrew/opt/graphviz"

# Install Python packages
python3 -m pip install \
--config-setting="--global-option=build_ext" \
--config-setting="--global-option=-I$(brew --prefix graphviz)/include/" \
--config-setting="--global-option=-L$(brew --prefix graphviz)/lib/" \
pygraphviz --break-system-packages

pip install pydot circuitgraph
```

## Building

1. Build Rust e-graph engine:
```bash
cd mig_egg
cargo install maturin
maturin develop # for Python
cargo build --release # for C++
cargo add flussab-aiger
```

2. Set environment variables:

If you need to have `cbc` first in your `PATH`, run:

```bash
echo 'export PATH="/opt/homebrew/opt/cbc/bin:$PATH"' >> ~/.zshrc
```

For compilers to find `cbc`, you may need to set:

```bash
export LDFLAGS="-L/opt/homebrew/opt/cbc/lib"
export CPPFLAGS="-I/opt/homebrew/opt/cbc/include"
```

For `pkg-config` to find `cbc`, you may need to set:

```bash
export PKG_CONFIG_PATH="/opt/homebrew/opt/cbc/lib/pkgconfig"
```


## Running Experiments
1. Build project:
```bash
./build.sh Release
```

2. Test extraction strategies:

```bash
cd mig_egg
RUSTFLAGS="-Awarnings" cargo test --no-default-features -- --nocapture
RUSTFLAGS="-Awarnings" cargo test --features ilp-cbc -- --nocapture
```

3. Run experiment:

```bash
nohup ./build/emig/emig 0 > emig_aig_k8.log 2>&1 &
nohup ./build/emig/emig 1 > emig_mig_k8.log 2>&1 &
```

## Benchmarks

The project includes various benchmark circuits from the EPFL benchmark suite:
- Basic: adder, arbiter, bar, cavlc
- Medium: c432, c499, c1355, c6288
- Complex: ctrl, dec, div, i2c, int2float

## Implementation Details

1. **Traditional Method (Baseline)**
- Implemented in C++ using mockturtle framework
- Uses conventional rewriting techniques

2. **E-graph Based Method**
- Uses equality saturation for optimization
- Implements multiple extraction strategies:
  - Greedy DAG extraction
  - Global greedy extraction
  - ILP-based extraction

3. **Optimization Process**
- Circuit parsing and MIG conversion
- K-cut decomposition
- E-graph transformation and optimization
- Result extraction and reconstruction

## Cite
``` bib
@inproceedings{fu2026elogic,
  author    = {Fu, Rongliang and Xuan, Wei and Yin, Shuo and Hu, Guangyu and Chen, Chen and Zhang, Hongce and Yu, Bei and Ho, Tsung-Yi},
  booktitle = {Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  title     = {{eLogic}: An E-Graph-based Logic Rewriting Framework for Majority-Inverter Graphs},
  year      = {2026},
  volume    = {},
  number    = {},
  pages     = {1-6},
  keywords  = {Logic synthesis; logic rewriting; majority-inverter graph; e-graph},
  doi       = {}
}
```

