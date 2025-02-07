## 🐍 Copperhead V2, - Columnar Parallel Pythonic framEwork for Run3 high-mass dilepton search

setup:
```bash
git clone https://github.com/aman0404/CMSAnalysisCoffea.git
cd CMSAnalysisCoffea
conda activate /depot/cms/kernels/root632 
```
If accessing datasets via `xRootD` will be needed:
```bash
source setup_proxy.sh
```

This would start the GRID certificate password prompt, and then once given the password, the conda env would be activated. Once voms proxy and conda env is activated, we can start on the tutorial.ipynb
