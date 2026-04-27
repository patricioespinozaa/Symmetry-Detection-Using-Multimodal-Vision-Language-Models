# Symmetry-Detection-Using-Multimodal-Vision-Language-Models


## Install

.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

.\venv\Scripts\python.exe -m pip install pytorch3d==0.7.8+pt2.8.0cu129 --extra-index-url https://miropsota.github.io/torch_packages_builder

## Dataset generation

### Download object and symmetries data

.... download from ...


### Generate rendered images

Execute:

python utils/data_render.py --input-folder data/objects/curated_axis_sym_obj --output-folder data/renders --illumination flat --repo-views 114 --image-size 224
