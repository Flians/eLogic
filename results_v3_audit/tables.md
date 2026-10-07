# ePMapping experiment report

Generated from `results_v3_audit` by `scripts/experiments/report.py`.

## extended tables (27 designs, README only -- the paper uses the 16 below)

## ISCAS subset (10 designs)

## tab:main vs ASPDAC'25 (its published Table 1 vs ours, A_e recipe)

| design | ASPDAC'25 area | ASPDAC'25 delay | ours area | ours delay | dA (%) | dD (%) |
| --- | --- | --- | --- | --- | --- | --- |
| aes_core | 1086.57 | 425.61 | 1027.88 | 410.72 | -5.40 | -3.50 |
| arbiter | 268.82 | 604.27 | 557.72 | 999.87 | +107.47 | +65.47 |
| div | 1176.29 | 60254.25 | 1184.08 | 52956.70 | +0.66 | -12.11 |
| ethernet | 3008.41 | 588.34 | 2540.68 | 540.76 | -15.55 | -8.09 |
| iwls05_i2c | 48.96 | 288.07 | 47.54 | 303.61 | -2.90 | +5.39 |
| mem_ctrl | 1799.15 | 1435.03 | 1885.47 | 1485.58 | +4.80 | +3.52 |
| RISC | 3139.86 | 1304.74 | 2920.58 | 1417.86 | -6.98 | +8.67 |
| sasc | 31.30 | 191.00 | 27.18 | 191.20 | -13.17 | +0.10 |
| simple_spi | 41.00 | 287.00 | 36.91 | 291.82 | -9.98 | +1.68 |
| spi | 164.10 | 489.49 | 149.35 | 480.72 | -8.99 | -1.79 |
| sqrt | 1110.02 | 78002.24 | 1105.68 | 82768.50 | -0.39 | +6.11 |
| systemcaes | 511.67 | 784.00 | 475.08 | 681.61 | -7.15 | -13.06 |
| systemcdes | 138.84 | 530.29 | 130.79 | 487.83 | -5.79 | -8.01 |
| tv80 | 351.11 | 907.86 | 339.01 | 846.12 | -3.44 | -6.80 |
| usb_funct | 695.57 | 702.87 | 645.16 | 663.36 | -7.25 | -5.62 |
| usb_phy | 23.63 | 163.82 | 22.54 | 158.68 | -4.61 | -3.14 |

**Means**: dA = +1.33%, dD = +1.80% (16 designs; negative = we are smaller/faster)

## tab:mfs3 (common serialized A_i input on 40 ASAP7 cells, one pass each)

Native parsed baseline accounting can differ through output buffers; each percentage uses its own measured start.

| design | ours A_i area | ours A_i delay | ABC A_i area | ABC A_i delay | mfs3 dA (%) | mfs3 dD (%) | mfs3 s | ours dA (%) | ours dD (%) | ours s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aes_core | 1113.13 | 410.49 | 1113.13 | 410.49 | -2.69 | +13.33 | 32 | -0.56 | -0.29 | 79 |
| arbiter | 557.72 | 999.87 | 557.72 | 999.87 | +0.00 | +0.00 | 96 | +0.00 | +0.00 | 16 |
| div | 3609.46 | 52450.60 | 3609.46 | 52450.16 | -- | -- | TO | -1.74 | -0.33 | 358 |
| ethernet | 3739.33 | 557.03 | 3739.47 | 557.03 | -0.65 | +10.19 | 52 | -0.67 | -3.26 | 110 |
| iwls05_i2c | 59.39 | 237.61 | 59.46 | 237.61 | -3.99 | -1.57 | 0 | -2.42 | +0.00 | 5 |
| mem_ctrl | 2323.92 | 1726.27 | 2323.99 | 1726.27 | -- | -- | TO | -1.63 | -0.89 | 130 |
| RISC | 3586.23 | 696.43 | 3594.28 | 696.43 | -2.85 | +68.75 | 92 | -1.24 | -5.06 | 132 |
| sasc | 35.85 | 140.61 | 35.85 | 140.61 | -1.84 | +0.00 | 0 | -1.34 | +0.00 | 4 |
| simple_spi | 49.64 | 214.50 | 49.64 | 214.50 | -2.70 | +0.00 | 0 | -0.48 | -4.54 | 5 |
| spi | 183.08 | 472.43 | 183.08 | 472.43 | -2.14 | +5.10 | 7 | -4.65 | -4.76 | 10 |
| sqrt | 1355.08 | 101970.00 | 1355.08 | 101970.38 | -- | -- | TO | -8.71 | -0.11 | 58 |
| systemcaes | 559.33 | 775.42 | 559.33 | 775.42 | -2.96 | +9.04 | 18 | -2.29 | -13.66 | 20 |
| systemcdes | 151.07 | 416.79 | 151.07 | 416.79 | -7.49 | +21.16 | 2 | -1.34 | -4.85 | 12 |
| tv80 | 462.39 | 799.99 | 462.39 | 799.99 | -6.99 | +17.57 | 38 | -2.60 | -4.29 | 27 |
| usb_funct | 801.30 | 421.38 | 802.42 | 421.38 | -2.55 | -0.32 | 1 | -2.81 | -4.61 | 28 |
| usb_phy | 27.15 | 155.17 | 27.15 | 155.17 | -9.65 | +0.00 | 1 | -3.28 | -5.70 | 5 |

**`mfs3 -a`**: mean dA = -3.58%, mean dD = +11.02% over the 13 of 16 it completes; **ePMapping**: mean dA = -2.24%, mean dD = -3.27% over all 16. ABC's numbers come from `tab_mfs3/abc.tsv`, its logs from `tab_mfs3/abc.<design>.log`.

On the same 13 completed designs, ePMapping has mean dA = -1.82% and dD = -3.92%. Ours s is whole-process wall time; ABC s excludes its separate CEC.

## fig:share data

### (a) layer funnel, one pass (L1 wins / L4 admitted / L4 wins)

| design | L1 wins | L4 admitted | L4 wins |
| --- | --- | --- | --- |
| c7552 | 5 | 442 | 53 |
| sin | 204 | 1548 | 14 |
| div | 918 | 3706 | 4 |
| multiplier | 1476 | 3783 | 175 |
| log2 | 1086 | 6305 | 75 |

    % L1 wins
    \addplot coordinates {(c7552,5) (sin,204) (div,918) (mult,1476) (log2,1086)}
    % L4 admitted
    \addplot coordinates {(c7552,442) (sin,1548) (div,3706) (mult,3783) (log2,6305)}
    % L4 wins
    \addplot coordinates {(c7552,53) (sin,14) (div,4) (mult,175) (log2,75)}

### e-graph memo cache (one pass)

| design | egg calls | cache hits | hit rate (%) |
| --- | --- | --- | --- |
| c7552 | 442 | 150 | 33.9 |
| sin | 1548 | 210 | 13.6 |
| div | 3706 | 1514 | 40.9 |
| multiplier | 3783 | 2508 | 66.3 |
| log2 | 6305 | 1831 | 29.0 |
| aes_core | 43335 | 37988 | 87.7 |
| arbiter | 905 | 889 | 98.2 |
| iwls05_i2c | 2036 | 1835 | 90.1 |
| mem_ctrl | 62553 | 51879 | 82.9 |
| sasc | 772 | 702 | 90.9 |
| simple_spi | 1091 | 974 | 89.3 |
| spi | 5280 | 4455 | 84.4 |
| systemcdes | 4974 | 4472 | 89.9 |
| tv80 | 11389 | 9706 | 85.2 |
| usb_phy | 1206 | 1095 | 90.8 |


### database coverage (`EPMAPPING_DB_STATS=1`)

| design | database | coverage (%) | best area > 0.2 (proxy) | avg best area |
| --- | --- | --- | --- | --- |
| c7552 | iwls.none.stats | 44.6 | 28296 | 0.390 |
| c7552 | ae.none.stats | 44.6 | 28296 | 0.390 |
| c7552 | ae.asap7_4in_2lvl.stats | 23.9 | 14702 | 0.382 |
| c7552 | iwls.asap7_4in_2lvl.stats | 23.9 | 14702 | 0.382 |
| c7552 | ae.asap7_4in_3lvl_a045.stats | 34.1 | 21416 | 0.363 |
| c7552 | iwls.asap7_4in_3lvl_a045.stats | 34.1 | 21416 | 0.363 |
| c7552 | ae.asap7_4in_3lvl_a050.stats | 44.6 | 28296 | 0.390 |
| c7552 | iwls.asap7_4in_3lvl_a050.stats | 44.6 | 28296 | 0.390 |
| c7552 | ae.asap7_4in_3lvl_a055.stats | 64.1 | 41036 | 0.430 |
| c7552 | iwls.asap7_4in_3lvl_a055.stats | 64.1 | 41036 | 0.430 |
| c7552 | ae.asap7_4in_3lvl_a060.stats | 79.1 | 50904 | 0.459 |
| c7552 | iwls.asap7_4in_3lvl_a060.stats | 79.1 | 50904 | 0.459 |


### (b) share of oracle time (11 designs complete in every arm, six rounds)

| stage | seconds | share (%) |
| --- | --- | --- |
| L1 single cell | 123.6 | 10.2 |
| L2 database | 49.5 | 4.1 |
| L3 synthesis | 418.6 | 34.5 |
| L4 e-graph | 597.2 | 49.2 |
| care mask | 25.8 | 2.1 |

Total 1214.8 s; cut enumeration, MFFC accounting and whole-window synthesis are outside this denominator.


### (c) signed area change when a stage is removed (11 designs complete in every arm)

| stage | area delta |
| --- | --- |
| L3 synthesis | +16.04 |
| L4 e-graph | +2.10 |
| L2 database | +0.00 |
| care mask | +4.09 |
| window | +12.52 |
These greedy-path marginal effects are not additive contributions.


## stage `tab_main`

| stage | design | base_a | out_a | dA% |
| --- | --- | --- | --- | --- |
| ae.r1 | arbiter | 557.72 | 557.72 | +0.00 |
| ae.r1 | iwls05_i2c | 47.66 | 47.6 | -0.13 |
| ae.r1 | div | 1184.5 | 1184.28 | -0.02 |
| ae.r1 | ethernet | 2541.41 | 2540.71 | -0.03 |
| ae.r1 | aes_core | 1029.98 | 1028.38 | -0.16 |
| ae.r1 | sasc | 27.28 | 27.18 | -0.37 |
| ae.r1 | simple_spi | 36.96 | 36.93 | -0.08 |
| ae.r1 | spi | 149.43 | 149.35 | -0.05 |
| ae.r1 | RISC | 2927.19 | 2921.42 | -0.20 |
| ae.r1 | mem_ctrl | 1895.4 | 1886.66 | -0.46 |
| ae.r1 | systemcdes | 131.06 | 130.79 | -0.21 |
| ae.r1 | systemcaes | 483.75 | 476.71 | -1.46 |
| ae.r1 | sqrt | 1157.38 | 1121.32 | -3.12 |
| ae.r1 | tv80 | 340.61 | 339.33 | -0.38 |
| ae.r1 | usb_phy | 22.54 | 22.54 | +0.00 |
| ae.r1 | usb_funct | 646.23 | 645.23 | -0.15 |
| ae | arbiter | 557.72 | 557.72 | +0.00 |
| ae | iwls05_i2c | 47.66 | 47.54 | -0.25 |
| ae | div | 1184.5 | 1184.08 | -0.04 |
| ae | aes_core | 1029.98 | 1027.88 | -0.20 |
| ae | ethernet | 2541.41 | 2540.68 | -0.03 |
| ae | sasc | 27.28 | 27.18 | -0.37 |
| ae | simple_spi | 36.96 | 36.91 | -0.14 |
| ae | spi | 149.43 | 149.35 | -0.05 |
| ae | systemcaes | 483.75 | 475.08 | -1.79 |
| ae | systemcdes | 131.06 | 130.79 | -0.21 |
| ae | mem_ctrl | 1895.4 | 1885.47 | -0.52 |
| ae | RISC | 2927.19 | 2920.58 | -0.23 |
| ae | tv80 | 340.61 | 339.01 | -0.47 |
| ae | usb_phy | 22.54 | 22.54 | +0.00 |
| ae | usb_funct | 646.23 | 645.16 | -0.17 |
| ai | arbiter | 557.72 | 557.72 | +0.00 |
| ae | sqrt | 1157.38 | 1105.68 | -4.47 |
| ai | iwls05_i2c | 58.77 | 57.56 | -2.06 |
| ai | aes_core | 1103.53 | 1099.71 | -0.35 |
| ai | ethernet | 3735.35 | 3712.69 | -0.61 |
| ai | sasc | 35.49 | 35.16 | -0.93 |
| ai | simple_spi | 49.18 | 48.93 | -0.51 |
| ai | spi | 182.17 | 173.65 | -4.68 |
| ai | mem_ctrl | 2278.85 | 2253.51 | -1.11 |
| ai | sqrt | 1354.92 | 1211.25 | -10.60 |
| ai | systemcaes | 558.96 | 545.97 | -2.32 |
| ai | RISC | 3564.94 | 3529.1 | -1.01 |
| ai | systemcdes | 150.27 | 148.36 | -1.27 |
| ai | div | 3609.3 | 3528.56 | -2.24 |
| ai | usb_phy | 27.15 | 26.41 | -2.73 |
| ai | tv80 | 460.2 | 449.52 | -2.32 |
| ai | usb_funct | 792.68 | 770.14 | -2.84 |

## stage `db`

| stage | design | base_a | out_a | dA% |
| --- | --- | --- | --- | --- |
| iwls.none.stats | c7552 | 80.37 | - | n/a |
| ae.none.stats | c7552 | 81.87 | - | n/a |
| ae.asap7_4in_2lvl.stats | c7552 | 81.87 | - | n/a |
| iwls.asap7_4in_2lvl.stats | c7552 | 80.37 | - | n/a |
| iwls.none | c7552 | 80.37 | 80.04 | -0.41 |
| iwls.asap7_4in_2lvl | c7552 | 80.37 | 80.01 | -0.45 |
| ae.asap7_4in_3lvl_a045.stats | c7552 | 81.87 | - | n/a |
| ae.none | c7552 | 81.87 | 81.64 | -0.28 |
| ae.asap7_4in_2lvl | c7552 | 81.87 | 81.65 | -0.27 |
| iwls.asap7_4in_3lvl_a045.stats | c7552 | 80.37 | - | n/a |
| ae.asap7_4in_3lvl_a050.stats | c7552 | 81.87 | - | n/a |
| iwls.asap7_4in_3lvl_a050.stats | c7552 | 80.37 | - | n/a |
| iwls.asap7_4in_3lvl_a045 | c7552 | 80.37 | 80.05 | -0.40 |
| ae.asap7_4in_3lvl_a045 | c7552 | 81.87 | 81.67 | -0.24 |
| ae.asap7_4in_3lvl_a055.stats | c7552 | 81.87 | - | n/a |
| ae.asap7_4in_3lvl_a050 | c7552 | 81.87 | 81.64 | -0.28 |
| iwls.asap7_4in_3lvl_a050 | c7552 | 80.37 | 80.04 | -0.41 |
| iwls.asap7_4in_3lvl_a055.stats | c7552 | 80.37 | - | n/a |
| ae.asap7_4in_3lvl_a060.stats | c7552 | 81.87 | - | n/a |
| iwls.asap7_4in_3lvl_a060.stats | c7552 | 80.37 | - | n/a |
| iwls.asap7_4in_3lvl_a055 | c7552 | 80.37 | 79.99 | -0.47 |
| ae.asap7_4in_3lvl_a055 | c7552 | 81.87 | 81.64 | -0.28 |
| ae.asap7_4in_3lvl_a060 | c7552 | 81.87 | 81.64 | -0.28 |
| iwls.asap7_4in_3lvl_a060 | c7552 | 80.37 | 80.01 | -0.45 |

## stage `funnel`

| stage | design | base_a | out_a | secs |
| --- | --- | --- | --- | --- |
| r1 | c7552 | 81.87 | 81.7 | 49 |
| r1 | sin | 276.16 | 275.57 | 72 |
| r1 | multiplier | 1270.83 | 1269.03 | 81 |
| r1 | div | 1184.5 | 1184.28 | 102 |
| r1 | log2 | 1462 | 1459.74 | 230 |

## stage `lever3`

| stage | design | base_a | out_a | dA% |
| --- | --- | --- | --- | --- |
| base | c7552 | 81.87 | 81.7 | -0.21 |
| iter5 | c7552 | 81.87 | 81.7 | -0.21 |
| iter10 | c7552 | 81.87 | 81.7 | -0.21 |
| node2000 | c7552 | 81.87 | 81.7 | -0.21 |
| base | c3540 | 41.88 | 41.68 | -0.48 |
| iter5 | c3540 | 41.88 | 41.68 | -0.48 |
| full | c7552 | 81.87 | 81.64 | -0.28 |
| iter10 | c3540 | 41.88 | 41.68 | -0.48 |
| node2000 | c3540 | 41.88 | 41.68 | -0.48 |
| full | c3540 | 41.88 | 41.66 | -0.53 |
| base | c5315 | 64.21 | 64.05 | -0.25 |
| iter5 | c5315 | 64.21 | 64.05 | -0.25 |
| iter10 | c5315 | 64.21 | 64.05 | -0.25 |
| node2000 | c5315 | 64.21 | 64.05 | -0.25 |
| full | c5315 | 64.21 | 63.84 | -0.58 |
| base | c2670 | 26.73 | 26.69 | -0.15 |
| iter5 | c2670 | 26.73 | 26.69 | -0.15 |
| iter10 | c2670 | 26.73 | 26.69 | -0.15 |
| node2000 | c2670 | 26.73 | 26.69 | -0.15 |
| full | c2670 | 26.73 | 26.69 | -0.15 |

## stage `ablation`

| stage | design | base_a | out_a | dA% |
| --- | --- | --- | --- | --- |
| noEGG | aes_core | 1029.98 | 1026.86 | -0.30 |
| noWINDOW | aes_core | 1029.98 | 1027.88 | -0.20 |
| noSYN | aes_core | 1029.98 | 1028.81 | -0.11 |
| full | arbiter | 557.72 | 557.72 | +0.00 |
| full | aes_core | 1029.98 | 1026.78 | -0.31 |
| noSYN | arbiter | 557.72 | 557.72 | +0.00 |
| noEGG | arbiter | 557.72 | 557.72 | +0.00 |
| noWINDOW | arbiter | 557.72 | 557.72 | +0.00 |
| noCARE | arbiter | 557.72 | 557.72 | +0.00 |
| noDB | arbiter | 557.72 | 557.72 | +0.00 |
| noSYN | div | 1184.5 | 1184.06 | -0.04 |
| full | div | 1184.5 | 1183.96 | -0.05 |
| noCARE | aes_core | 1029.98 | 1027.54 | -0.24 |
| noDB | aes_core | 1029.98 | 1026.78 | -0.31 |
| noWINDOW | div | 1184.5 | 1184.08 | -0.04 |
| noEGG | div | 1184.5 | 1184.17 | -0.03 |
| full | iwls05_i2c | 47.66 | 47.54 | -0.25 |
| noSYN | iwls05_i2c | 47.66 | 47.62 | -0.08 |
| noEGG | iwls05_i2c | 47.66 | 47.54 | -0.25 |
| noWINDOW | iwls05_i2c | 47.66 | 47.54 | -0.25 |
| noCARE | div | 1184.5 | 1184.02 | -0.04 |
| noCARE | iwls05_i2c | 47.66 | 47.55 | -0.23 |
| noDB | iwls05_i2c | 47.66 | 47.54 | -0.25 |
| noDB | div | 1184.5 | 1183.96 | -0.05 |
| noSYN | mem_ctrl | 1895.4 | 1887.23 | -0.43 |
| noWINDOW | mem_ctrl | 1895.4 | 1885.47 | -0.52 |
| noEGG | mem_ctrl | 1895.4 | 1875.98 | -1.02 |
| full | mem_ctrl | 1895.4 | 1874.78 | -1.09 |
| full | sasc | 27.28 | 27.18 | -0.37 |
| noWINDOW | sasc | 27.28 | 27.18 | -0.37 |
| noSYN | sasc | 27.28 | 27.17 | -0.40 |
| noEGG | sasc | 27.28 | 27.18 | -0.37 |
| noCARE | sasc | 27.28 | 27.18 | -0.37 |
| noDB | sasc | 27.28 | 27.18 | -0.37 |
| full | simple_spi | 36.96 | 36.89 | -0.19 |
| noSYN | simple_spi | 36.96 | 36.92 | -0.11 |
| noEGG | simple_spi | 36.96 | 36.89 | -0.19 |
| noWINDOW | simple_spi | 36.96 | 36.91 | -0.14 |
| noCARE | simple_spi | 36.96 | 36.85 | -0.30 |
| noDB | simple_spi | 36.96 | 36.89 | -0.19 |
| full | spi | 149.43 | 149.24 | -0.13 |
| noSYN | spi | 149.43 | 149.41 | -0.01 |
| noEGG | spi | 149.43 | 149.24 | -0.13 |
| noWINDOW | spi | 149.43 | 149.35 | -0.05 |
| noCARE | spi | 149.43 | 149.24 | -0.13 |
| noDB | spi | 149.43 | 149.24 | -0.13 |
| noCARE | mem_ctrl | 1895.4 | 1877.89 | -0.92 |
| full | systemcdes | 131.06 | 130.79 | -0.21 |
| noSYN | systemcdes | 131.06 | 130.98 | -0.06 |
| noEGG | systemcdes | 131.06 | 130.81 | -0.19 |
| noWINDOW | systemcdes | 131.06 | 130.79 | -0.21 |
| noCARE | systemcdes | 131.06 | 130.79 | -0.21 |
| noDB | systemcdes | 131.06 | 130.79 | -0.21 |
| noDB | mem_ctrl | 1895.4 | 1874.78 | -1.09 |
| full | tv80 | 340.61 | 338.63 | -0.58 |
| noSYN | tv80 | 340.61 | 339.63 | -0.29 |
| noWINDOW | tv80 | 340.61 | 339.01 | -0.47 |
| noEGG | tv80 | 340.61 | 339.22 | -0.41 |
| full | usb_phy | 22.54 | 22.44 | -0.44 |
| noSYN | usb_phy | 22.54 | 22.44 | -0.44 |
| noEGG | usb_phy | 22.54 | 22.44 | -0.44 |
| noWINDOW | usb_phy | 22.54 | 22.54 | +0.00 |
| noCARE | usb_phy | 22.54 | 22.44 | -0.44 |
| noCARE | tv80 | 340.61 | 338.82 | -0.53 |
| noDB | tv80 | 340.61 | 338.63 | -0.58 |
| noDB | usb_phy | 22.54 | 22.44 | -0.44 |

