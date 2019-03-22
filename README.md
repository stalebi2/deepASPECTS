# deepASPECTS
**Contributors:**  Salmonn Talebi, Rukhsana Yeasmin, and Tony Joseph

<p align="center">
<b>Abstract</b><br>
</p>
The  Alberta Stroke Program Early CT Score (ASPECTS) is widely used to assess early ischemic changes for stroke victims. Currently the process of reviewing ASPECT images is tedious and susceptible to human error. We propose a deep learning model to automatically classify patient’s CT scans with the correct aspect score. We use transfer learning with VGG16 to evaluate CNN models on 3 specific regions (M4, M5, and M6) of the brain. Two models were evaluated: a functional model that simultaneously predicts M4, M5, and M6 for the left and right side of the brain while the other had two sequential models which were used to independently evaluate 3 regions of left and right brain. Our best model achieves ~97.7% validation accuracy with a sensitivity of 0.79 and specificity of 0.98 on test data.


### **Introduction**
Evaluation of non-contrast CT of the patient's head is crucial to assess the severity of an ischemic stroke (stroke caused due to restricted or blocked blood flow). Alberta Stroke Program Early CT Score (ASPECTS) is a common medical standard used to communicate the severity of a stroke and to determine treatment options2. The score is based on the presence or absence of ischemia (blood flow blockage) on a non-contrast CT of the brain. There are 10 specific locations (Fig: 1A): (caudate (C), putamen (P), internal capsule (IC), insula (I), and 6 areas of the middle-cerebral artery territory (M1-M6) ) which are evaluated on each side of the brain to determine the score. ASPECT is scored for each side (R or L) out of 10 in which 1 point is deducted for each of the 10 locations affected. The regions are highlighted in Fig. (1). Calculated score is used to determine the treatment options and the prognosis.

![](https://imgur.com/79N2oAE.jpg)

### **Dataset and Features**
For a given patient we receive approximately 30 CT slices in an imaging format called DICOM. Each slice will represent a 5 mm horizontal slice between the neck and top of the head. Only a subset of the CTs (8-13 slices) will contain the regions of interest. An expert radiologist has provided 171 patient scans with labeled slices. Each slice is accompanied by a 20 set label for all 10 regions of interest split between left and right side of the brain. Final processed dataset contains ~1700 lower brain slices, of which ~900 had ischemia. We performed a (80%, 10%, 10%) split of the shuffled labeled data into training, dev and test set.  

**Data Pre-processing**
CT scans are measured in Hounsfield scale (HU), a measure of radiodensity. For a given patient CT scan, first we need to get slices in right order and then convert to HU unit5, 8. We can do this using information stored in the metadata: order slices using “InstanceNumber”, multiply by “RescaleSlope” and add “RescaleIntercept”. Next step is to filter out values outside the range of brain parenchyma, which typically ranges between -100 to 100 HU. However, to account for differences in CT scanners, we set the range from -200 to 200 HU. Any value outside this range has been set to the value of air (i.e., -1000 HU). Next, we normalize the image slices: values in the range -200 to 200 HU are scaled to 0.0 - 1.0, values below -200 HU are set to 0.0, and values higher than 200 HU are set to 1.0. Next step is to zero center all the images, where the mean is calculated from all the images of whole dataset. Fig. (2) shows sample processed slices.

![](https://imgur.com/HGIrmoZ.jpg)

### **Methods**
Given the limited amount of patient data we have (171 patients), it will be difficult to train a deep neural network from scratch. Hence, we apply transfer learning from pre-trained VGG1611 from Keras (github link). We have frozen all convolutional layers of VGG16 model and removed FC and dense layers. We experimented with 2 different types of models to train for lower part of the brain (M4, M5, M6), with frozen VGG16 as base:

**Sequential model:** Goal is to train one model for each side (left/ right) of the brains. For this model, a FC layer follows the output from VGG layer, followed by a 256-unit dense layer with RELU activation. A final 3-output dense layer node with sigmoid activation is used to predict the M4, M5, M6 regions.  

**Functional model:** A FC layer followed by a 256-node dense layer with RELU activation was inserted. A 2-output dense layer with softmax activation is connected to the 256-node dense layer to predict which side of the brain (left/ right) is impacted. Output from softmax layer is concatenated with 256-node dense layer, followed by a 6-unit dense layer with sigmoid activation to predict the M4, M5, M6 regions at both left and right side of brain. 

Batch normalization was performed after each dense layer explained above, which helped improve model performance significantly. We experimented with additional dense layers with dropout. However, model performance decreased with this setup, possibly due to limited data size. Since our slices are grayscale images, we applied the same slice to all 3 channels of the VGG16 input layer.

### **Experiments/Results/Discussion**
**Hyperparameter tuning:**
We evaluate the best set of hyperparameters (batch size, epochs, learning rate, and dense layer size) by using a random grid search.  Fig (3) shows validation accuracy for different combination of parameters. Final set of parameters were selected based on max validation accuracy (~ 97.5%): lr 0.001, batch size 32, number of epochs 20, 256 dense layer units.

![](https://imgur.com/SWKegdH.jpg)

**Performance Metrics:**
Keras reported accuracy was used to evaluate model performance during training. Fig. (4) shows accuracy and loss plots from Sequential and Functional models extracted from kears reported model train history.

![](https://imgur.com/OeKe4kM.jpg)

We had to be careful in selecting appropriate model performance metrics for stroke detection. It is critical that the model does not misclassify a healthy person as having stroke. Similarly model should detect all affected regions of brain accurately for a patient with stroke. Hence sensitivity and specificity are two important performance metrics we need to evaluate. Along with those, we used exact matching score and hamming loss reported by scikit-learn. We measure these metrics on a slice level instead of a patient level due to limited data. Below table summarizes performance scores on test data:

![](https://imgur.com/Fu9MQ8m.jpg)

### **Error Analysis:**
For most of the mispredicted slices, model prediction had partial matching with actual results, which resulted in comparatively lower sensitivity score. For some slices, multiple neighbor regions were impacted, but the model failed to detect all of those. In some other cases, the model picked more regions as impacted than actual results. In some rare cases, when a region is impacted there is a possibility of blood in nearby regions which might be low enough to be ignored by a doctor. Similarly scoring method may vary for different doctors. All of these will impact model performance. Old strokes may impact the performance of the model as well. These are not considered for ASPECT score calculation. However, model may fail to ignore these cases because of the trace of blood. Scans with motion (patient moved while taking the scan) or tilt also impact model performance, specifically it will impact the sensitivity. However we should be able to reduce this error with more data. 
With enough variation in the dataset model should be able to improve performance on all of these cases. To partially overcome the data issue, we applied careful augmentation. Considering human brain is symmetric for stroke detection, we applied mirroring on impacted slices, and flipped labels accordingly. This helped us improve model performance partially.

