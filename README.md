# ImageNet_Prepare

### Usage

**image_resizer.py** can read images and resize them.


**image_regenerate.py** can turn datasets stored in Cifar-10 style like ImageNet_32x32 into images.

### Example
**image_resizer.py**

```shell
python image_resizer.py -i ~/images/ILSVRC2015/Data/CLS-LOC/train -o ~/data/ -s 32 -a box -r -j 10 
python image_resizer.py -i ~/images/ILSVRC2015/Data/CLS-LOC/val -o ~/data/val -s 32 -a box
```
**image_regenerate.py**

modify directory variables in source code
```shell
python image_regenerate.py
```