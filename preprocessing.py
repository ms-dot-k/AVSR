import os, glob
import argparse, cv2
from src.data.landmark_transform import VideoProcess
from tqdm import tqdm
import numpy as np
import pickle
import shutil
from joblib import Parallel, delayed

def build_file_list(data_path, data_type):
    if data_type == 'LRS2':
        files = sorted(glob.glob(os.path.join(data_path, 'main', '*', '*.mp4')))
        files.extend(glob.glob(os.path.join(data_path, 'pretrain', '*', '*.mp4')))
    elif data_type == 'LRS3':
        files = sorted(glob.glob(os.path.join(data_path, 'trainval', '*', '*.mp4')))
        files.extend(glob.glob(os.path.join(data_path, 'pretrain', '*', '*.mp4')))
        files.extend(glob.glob(os.path.join(data_path, 'test', '*', '*.mp4')))
    else:
        raise NotImplementedError
    return [f.replace(data_path + '/', '')[:-4] for f in files]

def load_video(data_filename):
    """load_video.

    :param filename: str, the fileanme for a video sequence.
    """
    frames = []
    cap = cv2.VideoCapture(data_filename)                           
    while(cap.isOpened()):                                                       
        ret, frame = cap.read() # BGR                                            
        if ret:                                                                  
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames)


def per_file(f, args, video_process):
    save_path = os.path.join(args.save_path, 'Video', f)
    if os.path.exists(save_path + '.mp4'): return
    lm_save_path = os.path.join(args.save_path, 'Transformed_LM', f)
    aud_save_path = os.path.join(args.save_path, 'Audio', f)
    txt_save_path = os.path.join(args.save_path, 'Text', f)
    if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(lm_save_path)): os.makedirs(os.path.dirname(lm_save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(aud_save_path)): os.makedirs(os.path.dirname(aud_save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(txt_save_path)): os.makedirs(os.path.dirname(txt_save_path), exist_ok=True)
    if os.path.exists(os.path.join(args.landmark_path, f + '.pkl')):
        with open(os.path.join(args.landmark_path, f + '.pkl'), "rb") as pkl_file:
            lm = pickle.load(pkl_file)
        vid_name = os.path.join(args.data_path, f + '.mp4')
        vid = load_video(vid_name)

        if all(x is None for x in lm) or len(vid) == 0:
            return

        output = video_process(vid, lm)
        if output is None:
            return
        p_vid, yx_min, transformed_landmarks = output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(save_path + '.mp4', fourcc, 25, (96, 96))
        for v in p_vid:
            output.write(v)
        with open(lm_save_path + '.pkl', "wb") as pkl_file:
            pickle.dump({'landmarks': transformed_landmarks, 'yx_min': yx_min}, pkl_file)
        os.system(f'ffmpeg -loglevel panic -nostdin -y -i {vid_name} -acodec pcm_s16le -ar 16000 -ac 1 {aud_save_path}.wav')
        shutil.copy(vid_name[:-4] + '.txt', txt_save_path + '.txt')

def main():
    parser = get_parser()
    args = parser.parse_args()
    video_process = VideoProcess(mean_face_path='20words_mean_face.npy', convert_gray=False)
    file_lists = build_file_list(args.data_path, args.data_type)
    Parallel(n_jobs=3)(delayed(per_file)(f, args, video_process) for f in tqdm(file_lists))
    
def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for preprocessing."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="path including video and split files like train.txt"
    )
    parser.add_argument(
        "--landmark_path", type=str, required=True, help="path including landmark files"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="path for saving"
    )
    parser.add_argument(
        "--data_type", type=str, required=True, help="LRS2 or LRS3"
    )
    return parser


if __name__ == "__main__":
    main()