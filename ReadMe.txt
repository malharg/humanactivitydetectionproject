This is my opencv project there are various files in this folder the code.py contains my main code there are sample videos used for input there is a dataset file too and screenshots of my ouputs

For this project I have used ONNX (Open Neural Network Exchange) model. It's a ML model of trained models and is open source and the dataset i am using is "The Kinetics Human Action Video Dataset"  some actions i have separated down in Actions.txt file

Since this project involves image processing GPU is used. There is a deep learning implementation.

To run this project just open the terminal in this folder or gitbash can also work and type:

python code.py --model resnet-34_kinetics.onnx --classes Actions.txt --input "Your actual path (i have mentioned my paths in a text file)" --gpu 1 --output "output file name".mp4

Note:do not include inverted commas it's just for exmplaining what to enter

For accessing the webcam and determining what your action is type this in the terminal:

python code.py --model resnet-34_kinetics.onnx --classes Actions.txt

I have included my web cam output too

The model is not that accurate in samplevideo the baby was dancing but it's shown as doing laundry that maybe because the video is not that clear and the kitchen area in background actually somewhat looks like a laundry room allthough in sample2 the ouput is correct I have included all the screenshots of outputs as well.


Hope this helps looking forward to your answer.Thank you!