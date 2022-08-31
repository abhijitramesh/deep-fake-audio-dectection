import os
from PIL import Image
import numpy as np
import cv2
from linear_based import magnify_motion
from glob import glob

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split(input_path,num_split):
    im = Image.open(input_path)
    x_width, y_height = im.size
    split = np.int(x_width / 3)
    outputFileFormat = "{0}-{1}.png"
    baseName = input_path.replace(".png","") +"_cropped"
    edges = np.linspace(0, x_width, num_split+1)
    for i,(start, end) in enumerate(zip(edges[:-1], edges[1:])):
        box = (start, 0, end, y_height)
        a = im.crop(box)
        a.load()
        # open_cv_image = np.array(a) 
        # open_cv_image = open_cv_image[:, :, ::-1].copy() 
        outputName = os.path.join(outputFileFormat.format(baseName, i + 1))
        a.save(outputName, "png")

def generate_video(input_path):
    image_folder = '.' # make sure to use your folder
    baseName = input_path.replace(".png","")
    video_name = baseName+".mp4"
      
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png") and ("_cropped" in img)]
     

    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    video = cv2.VideoWriter(video_name, fourcc, 2, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
        # os.remove(image)
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

def video_to_spectogram(video_path):
    vidcap = cv2.VideoCapture(video_path)
    all_images = []
    success,image = vidcap.read()
    count = 0
    while success:
      all_images.append(image)
      success,image = vidcap.read()
      count += 1

    im_h = cv2.hconcat(all_images)

    cv2.imwrite(video_path.replace('.avi','')+'_motion_amplified.png', im_h)

def motion_amplification_pipeline(filename,freqs):
    
    spectogram = filename+'.png'
    
    print("Splitting spectogram to frames...")
    split(spectogram,len(freqs))
    
    print("Generating video from frames...")
    generate_video(spectogram)
    #input video path
    vidFname = filename+'.mp4'
    # the amplification factor
    factor = 20
    # low ideal filter
    lowFreq = 0.5
    # high ideal filter
    highFreq = 0.5
    # set the number of layers in laplacian pyramid
    levels = 2
    # set the type of filter used. Choices are 'butter' or 'ideal'
    filt = 'butter'
    # set the flag for using colored images. True -> colored, False -> grayscale
    rgb = True
    # output video path (always set it to .avi)
    vidFnameOut = filename+'.avi'
    
    print("Performing motion amplification on video...")
    magnify_motion(vidFname, vidFnameOut, lowFreq, highFreq, filt, levels, factor, rgb)
    
    print("Removing Orignal Video")
    os.remove(vidFname)

    video_to_spectogram(vidFnameOut)
    print("Removing Motion Amplified Video")
    os.remove(vidFnameOut)

def main():


    spoofed_data = glob("dataset/spoof/*")
    bonafied_data = glob("dataset/bonafide/*")

    for spectogram in spoofed_data:
        motion_amplification_pipeline(spectogram)
    for spectogram in bonafied_data:
        motion_amplification_pipeline(spectogram)

main()