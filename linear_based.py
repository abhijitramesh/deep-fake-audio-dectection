# source: https://github.com/ezioo2310/video_motion_magnification/blob/main/linear_based.py
import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack


#convert RBG to YIQ
def rgb2ntsc(src):
    [rows,cols]=src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#convert YIQ to RBG
def ntsc2rbg(src):
    [rows, cols] = src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    src_c=src.copy()
    height, width = src_c.shape[0:2]
    #we must have the rounded values in order to reconstruct the pyramid
    # assert height % 2**level == 0
    # assert width % 2**level == 0 
    if  height % 2**level != 0 or width % 2**level != 0:
        raise Exception('Height and width MUST be divisible by: 2 to the power of pyramid_levels!!')
    pyramid=[src_c]
    for i in range(level):
        src_c=cv2.pyrDown(src_c)
        pyramid.append(src_c)
    return pyramid

#Build Laplacian Pyramid
def build_laplacian_pyramid(src,levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid=[]
    for i in range(levels,0,-1):
        GE=cv2.pyrUp(gaussianPyramid[i])
        L=cv2.subtract(gaussianPyramid[i-1],GE)
        pyramid.append(L)
    return pyramid

#load video from file
def load_video(video_filename, RGB = True):
    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_tensor=np.zeros((frame_count,height,width,3) if RGB else (frame_count, height, width), dtype='int16')
    x=0

    while cap.isOpened():
        ret,frame=cap.read()
        if ret is True:
            video_tensor[x]=frame if RGB else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x+=1
        else:
            break
    return video_tensor,fps


# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i]=gaussian_frame
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification=50):
    return gaussian_vid*amplification

#reconstract video from original video and gaussian video
def reconstract_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=img+origin_video[i]
        final_video[i]=img
    return final_video

#save video to files
def save_video(video_tensor, name='out.avi', fps=30, RGB=True):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    num = 0 if RGB is False else 1 
    writer = cv2.VideoWriter(name, fourcc, fps, (width, height), num)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

#magnify color
def magnify_color(video_name,low,high,levels=3,amplification=20):
    t,f=load_video(video_name)
    gau_video=gaussian_video(t,levels=levels)
    filtered_tensor=temporal_ideal_filter(gau_video,low,high,f)
    amplified_video=amplify_video(filtered_tensor,amplification=amplification)
    final=reconstract_video(amplified_video,t,levels=3)
    save_video(final)

#build laplacian pyramid for video
def laplacian_video(video_tensor,levels=3, rgb = True):
    tensor_list=[]
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_laplacian_pyramid(frame,levels=levels)
        # import pdb; pdb.set_trace()
        if i==0:
            for k in range(levels):
                if rgb:
                    tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3), dtype='int16'))
                else:
                    tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1]), dtype='int16'))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list

#butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

#reconstract video from laplacian pyramid
def reconstract_from_tensorlist(filter_tensor_list,levels=3):
    final=np.zeros(filter_tensor_list[-1].shape, dtype='int16')
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]
        final[i]=up
    return final

#manify motion
def magnify_motion(video_name, video_name_output, low, high, filt = 'butter', levels=3, amplification=20, rgb = False):
    """
    low and high specify the corresponding frequencies. 
    filt specifies which temporal filter to use (choices are 'ideal' and 'butter'). 
    level specificies the number of levels in the pyramid and 
    amplification the amount of amplification of the frequency band determined by low and high. 
    rgb determines wheteher we use RGB images or grayscale.
    """

    t, f = load_video(video_name, RGB = rgb)
    lap_video_list=laplacian_video(t,levels=levels, rgb = rgb)
    #checking if the higher frequency is lower than fs/2 
    if high > f/2:
        raise Exception('Frequency band must be within 0 and fs/2 !!!')

    filter_tensor_list=[]
    for i in range(levels):
        if filt == 'butter':
            filter_tensor=butter_bandpass_filter(lap_video_list[i],low,high,f)
        elif filt == 'ideal':
            filter_tensor=temporal_ideal_filter(lap_video_list[i],low,high,f)
        else:
            raise Exception('If you want to use that filter, YOU code it!')
        filter_tensor*=amplification
        filter_tensor_list.append(filter_tensor.astype('int16'))
    recon=reconstract_from_tensorlist(filter_tensor_list, levels=levels)
    final=t+recon
    save_video(final, video_name_output, fps = f, RGB = rgb)

if __name__=="__main__":
    #magnify_color("baby.mp4",0.4,3)

    #input video path
    vidFname = "video/test.mp4"
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
    vidFnameOut =  './video_results/auto_linear_based_' + f'{lowFreq}-{highFreq}Hz_{levels}Levels_{factor}Amp_{filt}Filter.avi'
    print(vidFnameOut)
    magnify_motion(vidFname, vidFnameOut, lowFreq, highFreq, filt, levels, factor, rgb)