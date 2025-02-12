source /etc/profile.d/modules.sh

module --force purge

module load anaconda/2020.11

conda deactivate

source activate /depot/cms/kernels/coffea_latest
source /cvmfs/cms.cern.ch/cmsset_default.sh
voms-proxy-init --voms cms --valid 192:0:0
