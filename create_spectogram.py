import pandas as pd
import os
from glob import glob
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import concurrent.futures
import multiprocessing as mp


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_file_path(filename,file_list):
    return next((x for x in file_list if x.split("/")[-1].replace('.flac','')==filename), None)

def plot_time_series(data):
    plt.plot(data)

def plot_spectogram(data,filename):
    plt.specgram(data,cmap=plt.cm.binary)
    plt.axis('off')
    plt.savefig(filename+'.png',bbox_inches='tight',pad_inches=0)

def main():

    create_directory("dataset")
    create_directory("dataset/spoof")
    create_directory("dataset/bonafide")
    
    flac_path = os.path.join("ASVspoof2021_LA_eval","flac")
    flac_files = glob(flac_path+"/*")
    flac_filenames = [files.split("/")[-1].replace('.flac','') for files in flac_files]
    key_path = "keys/CM/trial_metadata.txt"
    df = pd.read_csv(key_path, sep="-",header=None)
    df.drop([0,2,3],axis=1,inplace=True)
    df.columns=["filename","label","type"]
    df = df[df['type']==' eval']
    df['label'] = df['label'].apply(lambda x: x.replace(" ",""))
    df = df[df.label=='bonafide']

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for i,vals in df.iterrows():
            vals.filename = vals.filename.replace(" ","")
            if(i%100==0):
                print(f"Plotting {i} / {len(df)}")
            
            if vals.filename in flac_filenames:
                data_path = os.path.join(flac_path,vals.filename.replace(" ","")+".flac")
                # print(f"Data Path = {data_path}")
                data,samplerate = sf.read(data_path)
            
            if vals.label=='spoof':
                path = os.path.join("dataset/spoof",vals.filename)
            else:
                path = os.path.join("dataset/bonafide",vals.filename)
            
            # print(f"Path = {path}")
            futures.append(executor.submit(plot_spectogram,data,path))

            if(i%3000==0):
                concurrent.futures.wait(futures)
                print("Executing total", len(futures), "jobs")
                for idx, future in enumerate(concurrent.futures.as_completed(futures, timeout=180.0)):
                    res = future.result()  # This will also raise any exceptions
                    print("Processed job", idx, "result", res)
                futures = []
        


main()