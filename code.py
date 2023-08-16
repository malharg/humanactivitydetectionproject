#importing the required packages
import numpy as np
import argparse
import imutils
import sys
import cv2

# constructing args parser to pass diff args.
argv = argparse.ArgumentParser()
argv.add_argument("-m", "--model", required=True, help="specify path to pre-trained model")
argv.add_argument("-c", "--classes", required=True, help="specify path to class labels file")
argv.add_argument("-i", "--input", type=str, default="", help="specify path to video file")
argv.add_argument("-o", "--output", type=str, default="",	help="path to output video file")
argv.add_argument("-d", "--display", type=int, default=1,	help="to display output frame or not")
argv.add_argument("-g", "--gpu", type=int, default=0,	help="whether or not it should use GPU")
args = vars(argv.parse_args())

# declare a variable to open and load contents of labels of activity .
ACT = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112       



print("Loading model For Human Activity Recognition")
gp = cv2.dnn.readNet(args["model"])



if args["gpu"] > 0:
	print("setting preferable backend and target to CUDA...")
	gp.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	gp.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Grab the pointer to the input video
print(" Accessing the video ...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = vs.get(cv2.CAP_PROP_FPS) 
print("FPS:", fps)


# Detect continoulsy till terminal is expilicitly closed 
while True:
    
    frames    = []  # frames for processing
    originals = []  # original frames


     
    for i in range(0, SAMPLE_DURATION):

        (grabbed, frame) = vs.read()
   
        if not grabbed:
            print("[INFO] No frame read from the stream - Exiting...")
            sys.exit(0)
 
        originals.append(frame)
        frame = imutils.resize(frame, width=400)
        frames.append(frame)
        

    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                  swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)


    #predicting activity
    gp.setInput(blob)
    outputs = gp.forward()
    label = ACT[np.argmax(outputs)]

    # for adding lables

    for frame in originals:
        
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
       	
     

        if args["display"] > 0:
            cv2.imshow("Activity Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
           
            if key == ord("q"):
                break

      
        if args["output"] != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')# *'MJPG' for .avi format
            writer = cv2.VideoWriter(args["output"], fourcc, fps,
                (frame.shape[1], frame.shape[0]), True)

        # write frame to ouput
        if writer is not None:
            writer.write(frame)
