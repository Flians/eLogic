
## requirement

- pygraphviz
- pydot
- circuitgraph

```bash
$ conda install -c conda-forge coincbc coin-or-cbc
$ sudo apt install coinor-libcbc-dev
$ sudo apt install zlib1g-dev yosys 
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
$ cargo install maturin
$ cd mig_egg && maturin develop
$ cargo add flussab-aiger
```
export GRAPHVIZ_DIR="/opt/homebrew/opt/graphviz"
python3 -m pip install \
--config-setting="--global-option=build_ext" \
--config-setting="--global-option=-I$(brew --prefix graphviz)/include/" \
--config-setting="--global-option=-L$(brew --prefix graphviz)/lib/" \
pygraphviz --break-system-packages

If you need to have cbc first in your PATH, run:
  echo 'export PATH="/opt/homebrew/opt/cbc/bin:$PATH"' >> ~/.zshrc

For compilers to find cbc you may need to set:
  export LDFLAGS="-L/opt/homebrew/opt/cbc/lib"
  export CPPFLAGS="-I/opt/homebrew/opt/cbc/include"

For pkg-config to find cbc you may need to set:
  export PKG_CONFIG_PATH="/opt/homebrew/opt/cbc/lib/pkgconfig"
  
## run

``` bash
(base) flynn@flynn-Precision-7920-Tower:~/workplace/MIGBalance$ nohup python3 -u mig/main.py --K 6 --obj 0 > egg_depth_K6.log 2>&1 &
[1] 23918
(base) flynn@flynn-Precision-7920-Tower:~/workplace/MIGBalance$ nohup python3 -u mig/main.py --K 6 --obj 1 > egg_area_K6.log 2>&1 &
[2] 24021
```