"""Charge the entire source-first frontend as mapping, including native search."""
from pathlib import Path
import run_with_mapping_budget as wrapper
_original=wrapper.mapping
wrapper.mapping=lambda argv: isinstance(argv,(list,tuple)) and len(argv)>0 and Path(str(argv[0])).name=='egraphmap_proportional.py' or _original(argv)
if __name__=='__main__':wrapper.main()
