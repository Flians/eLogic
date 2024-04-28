
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
```

## run

``` bash
(base) flynn@flynn-Precision-7920-Tower:~/workplace/MIGBalance$ nohup python3 -u mig/main.py --K 6 --obj 0 > egg_depth_K6.log 2>&1 &
[1] 23918
(base) flynn@flynn-Precision-7920-Tower:~/workplace/MIGBalance$ nohup python3 -u mig/main.py --K 6 --obj 1 > egg_area_K6.log 2>&1 &
[2] 24021
```