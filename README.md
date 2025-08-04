# S2AN-ASPM
Implementation code for "Spatial-semantic Attention Network with Adaptive Similarity Perception and Memory for Efficient Zero-shot Object-goal Visual Navigation".

## AI2-THOR Environment

### Setup

1. Create a environment using conda:
```
conda create -n S2AN_a python=3.8
conda activate S2AN_a
```

2. Clone the repository as:
```
git clone https://github.com/zihaodong/S2AN-ASPM.git
cd S2AN-ASPM/AI2-THOR
```

3. For the rest of dependencies, please run 
```
pip install -r requirements.txt 
```

### Data

Please follow the data structure and setup used in [Zero-Shot-Object-Navigation](https://github.com/pioneer-innovation/Zero-Shot-Object-Navigation).

Specifically:

- For **evaluation**, download [`data.zip` (~5 GB)](https://drive.google.com/drive/folders/1i6V_t6TqaTpUdUFpOJT3y3KraJjak-sa?usp=sharing), unzip it, and place it in the `S2AN-ASPM/AI2-THOR` folder.

- For **training**, download `train.zip` (~9 GB), unzip it, and move all `Floorplan*` folders into `./data/thor_v1_offline_data/`.

### Evaluation

Evaluate our model under 18/4 class split

```bash
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --model S2AN_ASPM \
    --gpu-ids 0 \
    --load_model trained_models/18_4_sota.dat \
    --split 18/4 \
    --zsd 1
```

Evaluate our model under 14/8 class split

```bash
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --model S2AN_ASPM \
    --gpu-ids 0 \
    --load_model trained_models/14_8_sota.dat \
    --split 14/8 \
    --zsd 1
```

### Training

Train our model under 18/4 class split

```bash
python main.py \
    --model S2AN_ASPM \
    --gpu-ids 0 \
    --workers 8 \
    --vis False \
    --zsd 1 \
    --partial_reward 1 \
    --split 18/4
```

Train our model under 14/8 class split

```bash
python main.py \
    --model S2AN_ASPM \
    --gpu-ids 0 \
    --workers 8 \
    --vis False \
    --zsd 1 \
    --partial_reward 1 \
    --split 14/8
```
### Visualization

The `visualization` folder contains a collection of scripts for visualizing various results.  



## HM3D Environment

### Setup

1. **Create Conda Environment**

```bash
conda create -n S2AN_h python=3.9
conda activate S2AN_h
```
2. Install habitat-sim

Follow the official installation instructions:
https://github.com/facebookresearch/habitat-sim#installation

3. Install habitat-lab

Follow the official installation instructions:
https://github.com/facebookresearch/habitat-lab

### Data

1. **Scene Dataset**  
Follow the instructions in the Habitat-Sim documentation to register and download the HM3D scene dataset:  
[https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d)

2. **Object Navigation Task Dataset**  
Visit the Habitat-Lab dataset page and locate the download link for **HM3DSem-v0.2**:  
[https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)

### Training

```bash
python -u -m  habitat_baselines.run --config-name=objectnav/ddppo_objectnav.yaml
```

### Evaluation

```bash
python -u -m  habitat_baselines.run --config-name=objectnav/ddppo_objectnav.yaml habitat_baselines.evaluate=True 
```
