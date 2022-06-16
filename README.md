# pure_transformer

pure_transformer is just a repo about transformer, derived from [SamLynnEvans/Transformer](https://github.com/SamLynnEvans/Transformer) and [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks)

1. reconstructe the project structure
2. update code with new pytorch api


# Usage

first, you would like to train your model, as follow
``` shell
python train.py -src_data pathto/lang1.txt -trg_data pathto/lang2.txt -src_lang lang1 -trg_lang lang2
```
after train model, you can get a vocab and a model in folder **weights/**.

we set deafult args in code. download this repo, and just click you run button. Then take a meal, whatever


### torchtext.data have no attribute 'Iterator' in the latest release
torchtext.data and torchtext.legacy will be complete removed.
In this [issue link](https://github.com/pytorch/text/issues/1275), you can find a 
[migration tutorial](https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb).

if you want to customize your data input, you can follow the tutorial, or you can like me, debug code, find what kind of data we put in model. 

### torch.autograd.Variable() deprecated
```
Variables are no longer necessary to use autograd with tensors. 
Autograd automatically supports Tensors with requires_grad set to True
```
here's my naive solution, substiting Variable() with torch.tensor()
