### Setup
```
# Step 1: create conda environment
conda create -n seg python==3.7.6
conda activate seg
conda install scikit-image jupyter
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
pip install git+https://github.com/geoopt/geoopt.git
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install lvis

# Step 2: Install detectron2
# Please clone my fork where I changed the roi_heads.py to make the MaskRCNN do class-agnostic segmentation.
git clone https://github.com/ZZWENG/detectron2.git
cd detectron2
git fetch origin
git checkout -b ca_seg origin/ca_seg  # the code change is on this branch.
cd ..
python -m pip install -e detectron2

# Step 3: Download datasets (COCO2017, COCO/LVIS annotation files)
# See required folder structure at https://github.com/facebookresearch/detectron2/tree/master/datasets
export DETECTRON2_DATASETS=path/to/datasets
```
### Train Class-agnostic Mask R-CNN
```python models/rpn.py```

### Train Embeddings
```python run.py```
