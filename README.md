## Abstract

This thesis investigates the domain of step detection using neural network models applied
to accelerometer and gyroscope data. The study investigates two different approaches
to step detection. The first approach employs Convolutional Neural Networks (CNNs)
to extract spatial features from time-series data, enabling the precise localization of step
events. Departing from previously used Long Short-Term Memory (LSTM) models,
this alternative demonstrates higher accuracy and remarkably improves inference speed.
The second approach adopts a YOLO-like architecture, aiming to predict steps in startend pairs, thus minimizing post-processing requirements. While the CNN-based models
exhibit superior accuracy compared to LSTM models, the YOLO-like model excels in
real-time processing efficiency. Traditionally, the accuracy of the models was described
by comparing the predicted number of steps to the original number of steps and the F1
score. This work also investigates how accurate the model is in predicting steps close to
the actual labels by calculating mean absolute error.
