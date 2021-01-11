# ASL-to-text-prototype
This project is a deep learning algorithm built in Keras, using the pretrained Inception V3 model to recognize and interpret images of the letters of the ASL alphabet into text.  
This project was inspired by, and uses the dataset from, the following project:  
	https://github.com/loicmarie/sign-language-alphabet-recognizer
  
<b>Approach and Methods</b>  
	The inputs for the model my algorithm builds (easy_train.py) are image data of a person’s hand forming the letters of the manual alphabet, and a few additional character inputs. The image directory  contains 29 subdirectories, originally containing 3000 images each, which I reduced to 500 images each by random sampling. The output easy_train is a trained Keras model.  
	The inputs for the image classification programs (easy_classify_larger.py and easy_evaluate_larger.py) are the trained Keras model, and either a specified image file for easy_classify, or an evaluation set of images for easy_evaluate. For easy_classify_webcam.py the inputs are the trained Keras model, and data from the computer’s webcam. For easy_evaluate the output is a text file containing the values generated by evaluation and prediction on the dataset. The output of easy_classify and easy_classify_webcam is to print the name of the category in which a given image is classified.  
	In easy_train, I used the Keras ImageDataGenerator functionality to import images from dataset in a given directory, organized into subdirectories by class, and create training and validation sets from this image data. Then, in the run_model() function I initialize the data generators, build a tensor on top of the InceptionV3 model, freeze the InceptionV3 convolutional layers and train on the top layers. After that, the InceptionV3 layers above 249 are unfrozen, the model is recompiled and trained again, and the final trained model is saved to the project directory.   
  
<b>Results and Conclusions</b>  
	Most of the time I spent on this project was on experimenting with Keras and InceptionV3, in order to get the model I was building to work in any capacity. It took quite a while to discover that InceptionV3 is not a sequential model, at which point I had to learn the Keras functional API, mostly by trial and error from the documentation’s limited instruction. By the time I was able to get a model working in any capacity, I only had time to do very limited testing of the resulting models.  
	Two versions of the programs for this project exist, one set which runs on images of size 150 x 150, and one on images of size 299 x 299. Most of my limited testing focused on comparing the performance of the models running on the two different size images. I randomly sampled the original subdirectories of 3000 images to create an evaluation set, different from the set used for training and validation, of 500 image subdirectories. Using the evaluate_generator method with a batch size of 100 images and 50 steps, I found that the results [loss, accuracy] were as follow when evaluating on the specified models:  
		ASLID_small4: [5.241470546722412, 0.432799996137619]  
		ASLID_large3: [2.327937116622925, 0.5425999957323074]  
	This clearly shows that the model using larger images functions significantly better overall, and so this model is the one used to process webcam image input.   
	If time permitted, I would have added alternate versions of several of the classes, to account for the fact that some letters can be correctly signed in multiple ways, and the dataset I used only accounts for one way for each letter. Additionally, I would have formatted the file output from easy_classify_webcam such that subsequent letters print on the same line, and input recognized as the “del” sign would delete the last printed character. I also would have added a sign to indicate a new line of text.  
  
<b>How to Run the Project</b>  
	There are two distinct aspects of this project which require different files to run, but all code and model files, plus the folder in which the dataset is organized, must be in the same directory in order for any of the code to work properly.  
Files needed:
1.	easy_train.py
2.	easy_classify_larger.py
3.	easy_evaluate_larger.py
4.	easy_classify_webcam.py
5.	ASLID_large3.model
6.	ASL_datasubset
7.	ASL_evalsubset  

To train a new model:
1.	Open easy_train.py
2.	In console, navigate to project directory.
3. 	Enter run_model()
4.	Output will be a file named ASLID.model

The given model can be tested using easy_classify_larger.py, easy_evaluate_larger.py, and easy_classify_webcam.py  
For easy_classify_larger.py:
1.	Specify directory image to be tested on.
		Ex: x = to_dir('ASL_datasubset/C/C107.jpg')
2.	Get model’s prediction for the specified image.
		Ex: predict_this(x)
For easy_evaluate_larger.py:
	run eval_model()
For easy_classify_webcam.py:
1.	Make sure the computer on which the program is running has a webcam and is running Windows, as this functionality may be OS dependent.
2.	Enter run_interpreter()
3.	A window with the name “Webcam” should open automatically.
4.	With the webcam window selected, place your hand in the frame while signing a letter, and when ready, hit the “z” key.
5.	Another window with the name “Capture” should open automatically, and in the console and associated text file, the predicted letter will be printed.
6.	Close the capture window.
7.	Repeat steps 4-6 as many times as desired, and when done, select the webcam window and press the escape key to end the program.
