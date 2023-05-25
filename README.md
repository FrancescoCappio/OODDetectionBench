# Out of distribution detection benchmark 

This repo contains a framework to execute a number of published and unpublished out of distribution detection methods on common and uncommon benchmarks. 
Implemented OOD detection methods can be divided into 2 groups:

 - methods requiring a finetuning on closed set data;
 - finetuning free methods, which simply compare support data (the closed set samples the others use
   for finetuning) and test data to provide normality scores for the latter. 

As a consequence this code can be used both to finetune an ImageNet pretrained model on a specific
OOD detection task, before evaluating a specific model, or to directly evaluate a pretrained model.

## Dependencies

The dependencies are listed in the `requirements.txt` file.

The pretrained models for CutMix, SimCLR and SupCLR, CSI, supCSI should be downloaded and put in the
`pretrained_models` directory. 
In order to enable easy results replication we collected the torch versions of all the needed models [here](https://drive.google.com/file/d/1zRxNO9uiZUdAwXYGA5EJkURobbD9dwOn/view?usp=sharing).

## Citation 

If you find this code useful, please cite our paper: 

```
@inproceedings{cappio2022relationalreasoning,
  title={Semantic Novelty Detection via Relational Reasoning},
  author={Francesco Cappio Borlino, Silvia Bucci, Tatiana Tommasi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
} 
```
