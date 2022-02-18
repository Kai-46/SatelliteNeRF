# note: launch this script via ". ./env.sh"

conda create -y -n SatelliteNeRF python=3.8 && conda activate SatelliteNeRF
pip install scipy numpy matplotlib opencv-python pyexr open3d tqdm icecream imageio imageio-ffmpeg
pip install trimesh pyquaternion
pip install tensorboardX
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch



