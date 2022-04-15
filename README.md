# Vegetable-Classifier

<center><img src= "https://raw.githubusercontent.com/ashwinshetgaonkar/kaggle-kernel-images/main/vegetables.jpg" alt ="vegetables" style='width:600px;'></center><br>


## Context:
* From vegetable production to delivery, several common steps are done manually like picking, and sorting vegetables.
* Therefore, it would be a great idea to automate this process in the coming future by using a robot empowering it using Computer Vision.


## Data:
The dataset contains three folders:
* train (15000 images)
* test (3000 images)
* validation (3000 images)
each of the above folders contains subfolders for different vegetables wherein the images for respective vegetables are present.
* There are images of 15 different vegetables.


## My work:
* My aim for this Project was to build a classification model that could provide every high accuracy greater than 95%.
* By using suitable augmentation layer and callbacks functions on pretained model(efficientnetB0), I managed to get an accuracy above 99%.
* I have further analysed and visualized the performance of the model.
* Deployed the project using flask and pywebio on heroku.
* Tech stack used: python, numpy, pandas, tensorflow, html, css, matplotlib, seaborn, pywebio, flask, heroku.


## App Working demo:
* The user has to first upload an image to classify,click the browse option to select the required image.
 <center><img src= "https://raw.githubusercontent.com/ashwinshetgaonkar/kaggle-kernel-images/main/vegetable-classifier-demo/vegetable_classifier_demo_1.PNG" alt ="vegetables" style='width:600px;'></center><br>



* After selecting click on the submit button.
 <center><img src= "https://raw.githubusercontent.com/ashwinshetgaonkar/kaggle-kernel-images/main/vegetable-classifier-demo/vegetable_classifier_demo_2.PNG" alt ="vegetables" style='width:600px;'></center><br>
 
 
 * Results:
  <center><img src= "https://raw.githubusercontent.com/ashwinshetgaonkar/kaggle-kernel-images/main/vegetable-classifier-demo/vegetable_classifier_demo_3.PNG" alt ="vegetables" style='width:600px;'></center><br>
  
   <center><img src= "https://raw.githubusercontent.com/ashwinshetgaonkar/kaggle-kernel-images/main/vegetable-classifier-demo/vegetable_classifier_demo_4.PNG" alt ="vegetables" style='width:600px;'></center><br>
   
   
 Note:Due to the space contrained I'm not able to provide deploy the app successfully on heroku but everything else is up to mark.
