# Autonomous Driving Challenge

![](videos/images_and_angles.gif)



 Your method should return a trained neural network class. An example class (from the Neural Networks module) is included in `sample_student.py`. Your submitted script should also inlcude a `NeuralNetwork()` class - you are free to modify this class or write your own from scratch. You'll also need to create a predict method:


````
def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given your trained neural network class, and an image filename, load image, make and return single predicted steering angle in degrees, as a float32. 
    '''
````

## Evalution 
Your code will be evaluated using `evaluate.py`, or a very similar variant. You can use this script locally to ensure you don't burn through your autograder submissions. Performance will be evaluated by comparing your predicted steering angles to human steering angles:

![](RMSE_Equation-01.png)


## The Data
Download training data [here](http://www.welchlabs.io/unccv/autonomous_driving/data/training.zip). 

## Packages
For this challenge you are only permitted to use numpy, opencv, tdqm, time, and scipy. Tensorflow and other high level ML tools are not permitted.

### Grading 

| RMSE (degrees)   | Points (10 max)  | 
| ------------- | ------------- | 
| RMSE <= 12.5     | 10  | 
| 12.5 < RMSE <= 15 | 9  |  
| 15 < RMSE <= 20 | 8  |   
| 20 < RMSE <= 25 | 7  |   
| 25 < RMSE <= 30 | 6  |   
| 30 < RMSE <= 40 | 5  |  
| RMSE > 40, or code fails to run | 4  |  





