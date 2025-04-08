# Data Preparing
## Synapse Data
1. Access to the synapse multi-organ dataset:
   1. Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
2. The directory structure of the whole project is as follows:

```bash
└── data
    └──Synapse
        ├── test_vol
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train
            ├── case0005_slice000.npz
            └── *.npz
```

