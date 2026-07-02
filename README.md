## TopoVST: Towards Topology-fidelitous Vessel Skeleton Tracking

This work is the code base for paper TopoVST: Towards Topology-fidelitous Vessel Skeleton Tracking(Under Review).

> By Yaoyu Liu, Minghui Zhang, Junjun He, Yun Gu
> > Institute of Medical Robotics, Shanghai Jiao Tong University, Shanghai, China.
>
> > Shanghai AI Lab, Shanghai, China.

A brief overview of our work. We utilize multi-scale GNNs for vessel skeleton direction and radius estimation.
![](assets/Introduction_v4_01.png)

### Training and testing

Before training TopoVST, you need to generate training samples. Run:
```bash
python src/scripts/data/generate_samples.py
```
NOTE that you need to specify path to your own dataset.

Training and testing are driven by CSV config files. Each row is one `run_id`
that pins dataset, tracker, hyper-parameters, and checkpoint paths:
```bash
python src/scripts/train/run_train.py --config-csv src/scripts/configs/train_configs.csv --run-id my_run --device cuda:0
python src/scripts/test/run_test.py  --config-csv src/scripts/configs/test_configs.csv  --run-id my_run --device cuda:0
```
Example CSV configs are coming soon.
