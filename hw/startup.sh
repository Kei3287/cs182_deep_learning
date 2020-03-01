conda remove --name cs182-assignment2 --all
conda create -n cs182-assignment2 python=3.6 pip
conda activate cs182-assignment2
pip install -r requirements.txt
conda install ipykernel
source activate cs182-assignment2
conda install -n cs182-assignment2 nb_conda_kernels
