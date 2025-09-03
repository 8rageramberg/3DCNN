# 3DCNN
Create a classifier of 3D models using a 3D convolutional neural network.


# setup

# 3D Model Classifier Setup

This project uses a conda environment with **PyTorch** and **PyTorch3D**.  
Follow these steps to reproduce the setup:

## 🔧 Environment Setup


# 1 Remove any old env named pytorch3d
conda deactivate || true
conda env remove -n pytorch3d -y

# 2 Make a fresh env
conda create -n pytorch3d python=3.10 -y
conda activate pytorch3d

# 3 PyTorch for macOS ARM (CPU/MPS build)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4 Build/runtime deps + Jupyter kernel
conda install -c conda-forge cmake ninja pybind11 eigen pkg-config llvm-openmp jupyter ipykernel -y
pip install iopath fvcore

# 5 Build & install PyTorch3D (no build isolation so it sees conda deps)
export MACOSX_DEPLOYMENT_TARGET=12.0 CC=clang CXX=clang++ ARCHFLAGS="-arch arm64"
pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 6 Register the kernel for Jupyter
python -m ipykernel install --user --name=pytorch3d --display-name "Python (pytorch3d)"

# 7 Sanity check (optional but recommended)

python - <<'PY'
import torch
print("Torch:", torch.__version__, "MPS avail:", torch.backends.mps.is_available())
import pytorch3d
print("PyTorch3D:", getattr(pytorch3d, "__version__", "installed"))
PY


# pandas, jupyter, matplotlib 
conda install -c conda-forge pandas jupyter -y
### this one didnt work: conda install -c conda-forge "numpy>=2.0" "matplotlib>=3.8" -y

conda install -c conda-forge "matplotlib>=3.8,<3.10" -y
