# Three-part constrained-area experiments

Every selected implementation satisfies its own initial area/delay bounds and has CEC evidence; native optimizer completions and independently validated deadline selections are counted separately.
ABC raw area-mode outputs are preserved. The guarded comparison reverts an infeasible completed output to its native initial mapping; it is an external output-selection policy, not an internal mfs3 timing constraint. ABC timeouts are excluded from the ABC-completed paired subset; our native timeout selections remain included as independently validated netlists under the declared selection policy.

{
  "whole_window_synthesis": false,
  "engine_cec_outputs": 348,
  "baseline_cec_outputs": 16,
  "main_own_area": -7.105879507980664,
  "main_own_delay": -7.432345784912146,
  "main_one_area": -5.042286058222667,
  "main_one_delay": -4.898327582414619,
  "main_other_area": -3.5862031015629836,
  "main_other_delay": -3.0039954046933905,
  "main_area_wins": 16,
  "reference_area": -9.11990081538116,
  "reference_delay": -6.923785422736665,
  "reference_wins": 14,
  "mfs_ours_area": -6.955472590948491,
  "mfs_ours_delay": -5.467522658158299,
  "abc_completed": 13,
  "abc_timeouts": 3,
  "abc_feasible": 6,
  "abc_delay_violations": 7,
  "cut_oracle_times": {
    "secs_stage1": 601.852282,
    "secs_stage2": 200.57857495899998,
    "secs_synth": 684.200633275,
    "secs_egg": 1511.19152864,
    "secs_care": 62.215049915,
    "secs_window": 0.0,
    "secs_chain": 0.28050365
  },
  "method_timeouts": 4,
  "selected_cec_outputs": 352,
  "phase_timed_full_runs": 16,
  "method_checkpoint_selections": 4,
  "full_main_ppa_equal": {
    "aes_core": true,
    "arbiter": true,
    "div": true,
    "ethernet": true,
    "iwls05_i2c": true,
    "mem_ctrl": true,
    "RISC": true,
    "sasc": true,
    "simple_spi": true,
    "spi": true,
    "sqrt": true,
    "systemcaes": true,
    "systemcdes": true,
    "tv80": true,
    "usb_funct": true,
    "usb_phy": true
  },
  "cofactor_activity": {
    "aes_core": {
      "cofactor_wins": 434,
      "secs_cofactor": 45.09395421,
      "available": true
    },
    "arbiter": {
      "cofactor_wins": 0,
      "secs_cofactor": 1.786226516,
      "available": true
    },
    "div": {
      "cofactor_wins": 170,
      "secs_cofactor": 14.82281307,
      "available": true
    },
    "ethernet": {
      "cofactor_wins": 9347,
      "secs_cofactor": 24.22869042,
      "available": true
    },
    "iwls05_i2c": {
      "cofactor_wins": 54,
      "secs_cofactor": 2.208414629,
      "available": true
    },
    "mem_ctrl": {
      "cofactor_wins": 2054,
      "secs_cofactor": 102.9946021,
      "available": true
    },
    "RISC": {
      "cofactor_wins": 4248,
      "secs_cofactor": 58.90694053,
      "available": true
    },
    "sasc": {
      "cofactor_wins": 54,
      "secs_cofactor": 0.67802588,
      "available": true
    },
    "simple_spi": {
      "cofactor_wins": 74,
      "secs_cofactor": 1.397267124,
      "available": true
    },
    "spi": {
      "cofactor_wins": 161,
      "secs_cofactor": 6.953816877,
      "available": true
    },
    "sqrt": {
      "cofactor_wins": 1266,
      "secs_cofactor": 23.5089005,
      "available": true
    },
    "systemcaes": {
      "cofactor_wins": 1248,
      "secs_cofactor": 17.00791783,
      "available": true
    },
    "systemcdes": {
      "cofactor_wins": 176,
      "secs_cofactor": 12.02843996,
      "available": true
    },
    "tv80": {
      "cofactor_wins": 310,
      "secs_cofactor": 19.67958414,
      "available": true
    },
    "usb_funct": {
      "cofactor_wins": 751,
      "secs_cofactor": 17.02384611,
      "available": true
    },
    "usb_phy": {
      "cofactor_wins": 24,
      "secs_cofactor": 0.964425752,
      "available": true
    }
  },
  "cofactor_seconds": 343.859906139,
  "abc_raw_area": -3.575691302200082,
  "abc_guarded_area": -1.5940348664441888,
  "paired_ours_area": -7.063179599648575,
  "abc_raw_delay": 11.018744353868293,
  "abc_guarded_delay": -0.14558063949823968,
  "paired_ours_delay": -6.610547597300915,
  "paired_area_wins": 7,
  "guarded_paired_area_wins": 11,
  "ablation": {
    "full": {
      "area": 0.0,
      "worse": 0,
      "better": 0,
      "own_area": -7.105879507980664,
      "own_delay": -7.432345784912146,
      "optimizer_seconds": 3179.24647139,
      "time_ratio": 1.0
    },
    "noSYN": {
      "area": 394.70999999999947,
      "worse": 16,
      "better": 0,
      "own_area": -3.9051680184811115,
      "own_delay": -5.61044267703323,
      "optimizer_seconds": 2320.9129982329996,
      "time_ratio": 0.7300198393294975
    },
    "noEGG": {
      "area": 3.8899999999995742,
      "worse": 11,
      "better": 0,
      "own_area": -7.0514191741609835,
      "own_delay": -7.423816406565098,
      "optimizer_seconds": 1633.2478376849997,
      "time_ratio": 0.5137216797699005
    },
    "noDB": {
      "area": 1.6199999999993686,
      "worse": 5,
      "better": 7,
      "own_area": -7.093108356692806,
      "own_delay": -7.260743611263597,
      "optimizer_seconds": 2609.691211456,
      "time_ratio": 0.8208521217026045
    },
    "noCARE": {
      "area": 15.159999999999123,
      "worse": 13,
      "better": 1,
      "own_area": -7.0173015738841995,
      "own_delay": -7.242745072389706,
      "optimizer_seconds": 3007.215124551,
      "time_ratio": 0.9458892701817528
    },
    "noCHAIN": {
      "area": 334.08000000000004,
      "worse": 1,
      "better": 0,
      "own_area": -3.3620654077152974,
      "own_delay": -2.816245691900054,
      "optimizer_seconds": 3175.8588811349996,
      "time_ratio": 0.998934467558434
    },
    "noSHARE": {
      "area": 334.08000000000004,
      "worse": 1,
      "better": 0,
      "own_area": -3.3620654077152974,
      "own_delay": -2.816245691900054,
      "optimizer_seconds": 3163.902487606,
      "time_ratio": 0.9951737042339811
    },
    "noNARROW": {
      "area": 0.9199999999998454,
      "worse": 6,
      "better": 0,
      "own_area": -7.102617055187663,
      "own_delay": -7.428114490401176,
      "optimizer_seconds": 3641.244169993,
      "time_ratio": 1.145316729218861
    },
    "noPHASE": {
      "area": 243.8199999999998,
      "worse": 15,
      "better": 0,
      "own_area": -5.032886323499387,
      "own_delay": -6.541978030754591,
      "optimizer_seconds": 2806.82172616,
      "time_ratio": 0.8828575423197145
    },
    "noSHANNON": {
      "area": 148.1099999999998,
      "worse": 15,
      "better": 0,
      "own_area": -5.756875883151738,
      "own_delay": -7.533658354410004,
      "optimizer_seconds": 3090.774741638,
      "time_ratio": 0.9721721072750552
    },
    "noDAVIO": {
      "area": 39.51999999999929,
      "worse": 13,
      "better": 1,
      "own_area": -6.712804651724133,
      "own_delay": -7.085329215076569,
      "optimizer_seconds": 3078.327531857,
      "time_ratio": 0.9682569626352759
    },
    "noREUSE": {
      "area": 231.81999999999917,
      "worse": 15,
      "better": 0,
      "own_area": -5.11349455272647,
      "own_delay": -6.705610925291919,
      "optimizer_seconds": 3205.44880616,
      "time_ratio": 1.00824168085293
    },
    "noPRICE": {
      "area": 163.5899999999991,
      "worse": 15,
      "better": 0,
      "own_area": -5.793188516806176,
      "own_delay": -6.540764749938204,
      "optimizer_seconds": 2592.1153882420003,
      "time_ratio": 0.8153238232922219
    },
    "cache64K": {
      "area": 0.2799999999999727,
      "worse": 1,
      "better": 0,
      "own_area": -7.104956220020338,
      "own_delay": -7.432345784912146,
      "optimizer_seconds": 3747.165722965,
      "time_ratio": 1.178633288323412
    },
    "noAIGREUSE": {
      "area": 0.0,
      "worse": 0,
      "better": 0,
      "own_area": -7.105879507980664,
      "own_delay": -7.432345784912146,
      "optimizer_seconds": 3194.0073311959995,
      "time_ratio": 1.0046428799839309
    },
    "noLIBSHARE": {
      "area": 0.0,
      "worse": 0,
      "better": 0,
      "own_area": -7.105879507980664,
      "own_delay": -7.432345784912146,
      "optimizer_seconds": 3270.603520607999,
      "time_ratio": 1.028735440941783
    },
    "noQUERY": {
      "area": 69.75999999999976,
      "worse": 2,
      "better": 0,
      "own_area": -6.729255608150609,
      "own_delay": -6.959253236038747,
      "optimizer_seconds": 5321.076410435999,
      "time_ratio": 1.6736910643198948
    },
    "noTRUTH": {
      "area": 0.0,
      "worse": 0,
      "better": 0,
      "own_area": -7.105879507980664,
      "own_delay": -7.432345784912146,
      "optimizer_seconds": 3246.8301071999995,
      "time_ratio": 1.0212577528726332
    },
    "allPIVOTS": {
      "area": -36.93000000000009,
      "worse": 1,
      "better": 10,
      "own_area": -7.833302625486684,
      "own_delay": -7.228076584473621,
      "optimizer_seconds": 4062.988971702,
      "time_ratio": 1.2779723145924005
    }
  },
  "chain_activity": {
    "aes_core": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "arbiter": {
      "chain_wins": 384,
      "secs_chain": 0.053694636,
      "available": true
    },
    "div": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "ethernet": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "iwls05_i2c": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "mem_ctrl": {
      "chain_wins": 0,
      "secs_chain": 0.240288323,
      "available": true
    },
    "RISC": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "sasc": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "simple_spi": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "spi": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "sqrt": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "systemcaes": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "systemcdes": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "tv80": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "usb_funct": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    },
    "usb_phy": {
      "chain_wins": 0,
      "secs_chain": 0.0,
      "available": true
    }
  }
}

## Per-design fully preprocessed results

| Design | A0 | D0 | A | D | dA % | dD % |
| --- | --- | --- | --- | --- | --- | --- |
| aes_core | 1029.98 | 410.72 | 996.04 | 407.66 | -3.30 | -0.75 |
| arbiter | 557.72 | 999.87 | 223.64 | 261.39 | -59.90 | -73.86 |
| div | 1184.5 | 53105.06 | 1182.38 | 53024.38 | -0.18 | -0.15 |
| ethernet | 2541.41 | 543.88 | 2531.03 | 543.56 | -0.41 | -0.06 |
| iwls05_i2c | 47.66 | 303.61 | 45.11 | 290.02 | -5.35 | -4.48 |
| mem_ctrl | 1895.4 | 1519.75 | 1818.16 | 1461.58 | -4.08 | -3.83 |
| RISC | 2927.19 | 1453.77 | 2807.4 | 1392.91 | -4.09 | -4.19 |
| sasc | 27.28 | 191.2 | 26.96 | 191.2 | -1.17 | +0.00 |
| simple_spi | 36.96 | 308.22 | 35.89 | 293.36 | -2.90 | -4.82 |
| spi | 149.43 | 500.98 | 146.68 | 450.9 | -1.84 | -10.00 |
| sqrt | 1157.38 | 88015.07 | 1071.21 | 80961.45 | -7.45 | -8.01 |
| systemcaes | 483.75 | 709.03 | 447.57 | 682.83 | -7.48 | -3.70 |
| systemcdes | 131.06 | 488.85 | 125.89 | 487.52 | -3.94 | -0.27 |
| tv80 | 340.61 | 846.12 | 330.87 | 833.94 | -2.86 | -1.44 |
| usb_funct | 646.23 | 668.08 | 622.62 | 664.39 | -3.65 | -0.55 |
| usb_phy | 22.54 | 163.29 | 21.39 | 158.68 | -5.10 | -2.82 |
