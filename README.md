# deepASPECTS
**Contributors:**  Rukhsana Yeasmin, Salmonn Talebi, and Tony Joseph

<p align="center">
<b>Abstract</b><br>
</p>
The  Alberta Stroke Program Early CT Score (ASPECTS) is widely used to assess early ischemic changes for stroke victims. Currently the process of reviewing ASPECT images is tedious and susceptible to human error. We propose a deep learning model to automatically classify patient’s CT scans with the correct aspect score. We use transfer learning with VGG16 to evaluate CNN models on 3 specific regions (M4, M5, and M6) of the brain. Two models were evaluated: a functional model that simultaneously predicts M4, M5, and M6 for the left and right side of the brain while the other had two sequential models which were used to independently evaluate 3 regions of left and right brain. Our best model achieves ~97.7% validation accuracy with a sensitivity of 0.79 and specificity of 0.98 on test data.
