# D-JNDQ
Public repository for D-JNDQ: Learning Image Quality from Just Noticeable Differences


## Training scripts

* *TrainingLoop_MCL-JCI.py* contains the necessary functions to retrain the defined model on MCL-JCI Dataset
* *DataLoader_MCLJCI.py* is required to load the data during training
* *SiameseNet.py* is required to define the model architecture.

## Pre-trained model parameters

* Pretrained model parameters with the optimal hyper parameters can be found in *BestModelParams.pt* file.
* Usage example can be found in [TID-2013](http://www.ponomarenko.info/tid2013.htm) [[1]](#1) evaluation script. (*Eval_TID2013.py*)
* Note that, input image pairs first need to be pre-processed with [HDR-VDP-3](https://sourceforge.net/projects/hdrvdp/files/hdrvdp/).
* Example matlab code for pre-processing stage is given in the following section

## Pre-processing input images with HDR-VDP-3 [[2]](#2)


By default, viewing density and display device parameters may not be given with the dataset. 
Although, they are not difficult to find out from the dataset papers, default values below can be adapted
in the case where they are not known.

note: A function is provided to calculate ppd value in HDR-VDP-3. 
However here is another example [link](http://phrogz.net/tmp/ScreenDensityCalculator.html#find:density,pxW:1920,pxH:720,size:12.3,sizeUnit:in,axis:diag,distance:31,distUnit:in) to calculate viewing density


By default, HDR-VDP 3 does not provide Achromatic responses as the main function output.
A small addition after lines 484 and 485 in main HDR-VDP 3 function will allow to access achromatic responses 
produced by the model.

```
# line 484-485
[B_R, L_adapt_reference, bb_padvalue, P_ref] = hdrvdp_visual_pathway( reference, 'reference', metric_par, -1 );
[B_T, L_adapt_test, bb_padvalue, P_test] = hdrvdp_visual_pathway( test, 'test', metric_par, bb_padvalue );

# additional lines:
res.P_ref = P_ref;
res.P_test = P_test;
```

We have used the above lines to extract Achromatic responses with main function call.

After addition of the lines above, script below can be used to acquire Achromatic responses, as well as the HDR-VDP 3 predictions.


```
# initiate display device parameters:
PeakLum = 300;
BlackLvl = 0.1;
# initiate viewing density parameter. 
ppd = 90;

# Read reference and distorted images.
ref_img = imread(ref_image_path)
dist_img = imread(distorted_image_path)

# Convert images into Luminance values (cd/m^2) based on display device parameters
LumaRef = 0.2126.*refimg(:,:,1)./255+ 0.7152.*refimg(:,:,2)./255+ 0.0722.*refimg(:,:,3)./255;
LumiRef = (PeakLum-BL)*(LumaRef.^(2.2))+BL;
LumaDist = 0.2126.*distimg(:,:,1)./255+ 0.7152.*distimg(:,:,2)./255+ 0.0722.*distimg(:,:,3)./255;
LumiDist = (PeakLum-BL).*(LumaDist.^(2.2))+BL;

# 
out = hdrvdp3('detection', LumiDist, LumiRef, 'luminance', ppd);
ref_AchromaticResponse = sss.P_ref;
dist_AchromaticResponse = sss.P_test;
```

## Evaluation scripts

* *Eval_TID2013.py* scripts contains the evaluation loop over TID-2013 dataset 
* It is currently implemented to work on GPU, however it is straightforward to load the same model for CPU calculation.

## Calculated D-JNDQ scores for TID-2103 images
* Predicted similarity scores with D-JNDQ on TID-2013 dataset can be found on *D-JNDQ_withfnames.csv* file.
* Additionally, predicted scores are also saved as a txt file without filenames in order to be directly input into 
  executable evaluation functions provided by TID-2013 dataset. They can be found in *D-JNDQ.txt* file.
  

## Link to pre-processed MCL-JCI dataset images

* A link will be added soon

## Link to pre-processed TID-2013 dataset images

* A link will be added soon


## References
<a id="1">[1]</a> 
Rafal Mantiuk, Kil Joong Kim, Allan G. Rempel, and Wolfgang  Heidrich,   
“Hdr-vdp-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions”
ACM Trans. Graph., vol. 30, no. 4,July 2011.

<a id="1">[2]</a> 
Nikolay Ponomarenko, Lina Jin, Oleg Ieremeiev, Vladimir Lukin, Karen Egiazarian, Jaakko Astola, Benoit Vozel, Kacem Chehdi, Marco Carli, Federica Battisti, and C.-C. Jay Kuo,  
“Image database tid2013: Peculiarities, Results and Perspectives”
Signal Process-ing: Image Communication, vol. 30, pp. 57 – 77, 2015
