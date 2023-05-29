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

We downloaded the public models for resnet101 versions of CutMix, SimCLR, SupCLR, CSI, supCSI from
the original repositories and converted them to make them compatible with our framework. 
Converted models can be found
[here](https://drive.google.com/file/d/1w41RjKaOx5tbOcb3AleAWAOTzNxEw9ap/view?usp=sharing)
Downloaded models should be put in `pretrained_models` directory. 

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
