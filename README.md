# Motion-compensated MR CINE reconstruction with reconstruction-driven motion estimation
[Jiazhen Pan](https://jzpeterpan.github.io/), 
[Wenqi Huang](https://scholar.google.com/citations?user=to2zNj4AAAAJ&hl=de), 
[Daniel Rueckert](https://scholar.google.com/citations?user=H0O0WnQAAAAJ&hl=en), 
[Thomas Kuestner](https://www.medizin.uni-tuebingen.de/de/das-klinikum/mitarbeiter/profil/252),
and 
[Kerstin Hammernik](https://scholar.google.com/citations?user=IIqyUmAAAAAJ&hl=de)

**[Paper Link](https://ieeexplore.ieee.org/abstract/document/10436641)**

Conventional motion-compensated MR reconstruction (MCMR) methods rely on two-stages processes, i.e. separate motion estimation and reconstruction.
This scheme is inefficient and the motion estimation itself (can be aliasing-corrupted) is not optimized for the reconstruction task. 
The goal of motion estimation and reconstruction is not aligned, which can lead to suboptimal results.
In this work, we propose a reconstruction-driven motion estimation method and unify the motion estimation and reconstruction into a single end-to-end learning framework.
The proposed method is evaluated on private and also public ([OCMR](https://www.ocmr.info/)) cardiac cine dataset. The results 
show that the proposed method outperforms the conventional two-stage MCMR methods and also non-MCMR methods in both quantitative and qualitative evaluations.

## Setup and Installation
```
pip install -r requirements.txt
``` 
Pytorch version has to be >= 1.9.0.

Furthermore, you would need to install [splatting function](https://github.com/hperrot/splatting?tab=readme-ov-file) for the forward motion warping.

## Experiments
To run the demo training and testing, you can use the following commands:
```python
python demo.py 
```
The configurations are defined in the `MocoRecon.yaml` file.

It is noting that we do not provide the CINE data for training. In the demo code we use dummy (random generated) data
to simulate the training process. You should replace the data with your own dataloader, either private or public dataset e.g. [OCMR](https://www.ocmr.info/).

## References
```
@inproceedings{Pan_TMI_2024,
     author = {Pan, Jiazhen and Huang, Wenqi and Rueckert, Daniel and Kuestner, Thomas and Hammernik, Kerstin},
     title = {Motion-compensated MR CINE reconstruction with reconstruction-driven motion estimation},
     booktitle = {IEEE Transactions on Medical Imaging},
     year = {2024}
     }
```
## Acknowledgement
The code of this repository is developed based on [RAFT](https://github.com/princeton-vl/RAFT) and [merlin](https://github.com/midas-tum/merlin).
Shout out to the authors for their great work.


