import skvideo.io
import skimage.transform
import numpy as np
import glob
import os
import csv

def label_num(clip_label):
    if clip_label == 'SIL':
        return 0
    clip_label = clip_label.split('_')[0]
    if clip_label == 'pour':
        return 1
    if clip_label == 'cut':
        return 2
    if clip_label == 'crack':
        return 3
    if clip_label == 'take':
        return 4
    if clip_label == 'put':
        return 5
    if clip_label == 'add':
        return 6
    if clip_label == 'stir':
        return 7
    if clip_label == 'fry':
        return 8
    if clip_label == 'butter':
        return 9
    if clip_label == 'stirfry':
        return 10
    if clip_label == 'peel':
        return 11
    if clip_label == 'squeeze':
        return 12
    if clip_label == 'spoon':
        return 13
    if clip_label == 'smear':
        return 14

def is_stereo(person, list):
    for cam in list:
        if cam == person + '/stereo':
            return True
    return False


dire = '/media/fj-sanguino/Elements/Breakfast/BreakfastII_15fps_qvga_sync/'
people = sorted(glob.glob(dire + '*'))

split = [people[0:12], people[13:25], people[26:38], people[39:52]]

for sp in range(len(split)):
    spli = split[sp]
    print('---------------------SPLIT ' + str(sp) + '---------------------')
    out_dir = '/media/fj-sanguino/Elements/Breakfast/Splits/split' + str(sp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    csv_file = os.path.join(out_dir, 'labels_split' + str(sp) + '.csv')
    print(csv_file)
    with open(csv_file, mode='w') as f:
        employee_writer = csv.writer(f)
        employee_writer.writerow(['video_index', 'video_name', 'clip_start', 'clip_end','clip_name', 'label_name', 'label_number'])
    video_index = 0
    for person in spli:
        print('------------' + person[person.rfind('/')+1:] + '------------' )
        cams = sorted(glob.glob(person + '/*'))
        #print(cams)
        if is_stereo(person, cams):
            cams.remove(cams[cams.index(person + '/stereo')])#removes stereo
        #print(cams)
        for cam in cams:
            print('------' + cam[cam.rfind('/', 0, cam.rfind('/')) + 1: ] + '------')

            labels = sorted(glob.glob(cam + '/*.labels'))
            #print(len(labels))
            videos = [label[:label.rfind('.')]for label in labels]
            #print(len(videos))
            for vid_Idx in range(len(labels)):
                label = labels[vid_Idx]
                video = videos[vid_Idx]
                print('---' + video[video.rfind('/')+1:video.rfind('.')] + '---' )
                f = open(label, "r")
                clips=[]
                for x in f:
                    clips.append(x[:-2].split(' '))
                #print(clips)


                videogen =skvideo.io.vreader(video)
                frames = []
                i = 0
                for frameIdx, frame in enumerate(videogen):
                    init_frame, end_frame = [int(x) for x in clips[i][0].split('-')]
                    clip_label = clips[i][1]
                    #print(init_frame, end_frame, frameIdx)
                    frames.append(frame)
                    if frameIdx + 1 >= end_frame:
                        #print(frameIdx)
                        i = i + 1
                        frames = np.asarray(frames)
                        frames = frames.astype(np.uint8)
                        #print(frames.shape)
                        cam_only = video[video.rfind('/', 0, video.rfind('/'))+1: video.rfind('/')]
                        video_only = video[video.rfind('/')+1:video.rfind('.')]
                        output_file = os.path.join(out_dir, video_only + '_' + cam_only + '-' + str(i))
                        #print(output_file, label_num(clip_label))
                        skvideo.io.vwrite(output_file + '.avi', frames)
                        #2print(frames.shape)
                        fields = [video_index, video[video.rfind('/', 0, video.rfind('/', 0, video.rfind('/')))+1:],
                                  init_frame, end_frame, output_file[output_file.rfind('/', 0, output_file.rfind('/'))+1:] + '.avi', clip_label, label_num(clip_label)]
                        video_index = video_index + 1
                        with open(csv_file, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(fields)
                        frames = []
                #init_frame, end_frame = [int(x) for x in clips[i][0].split('-')]
                #clip_label = clips[i][1]
                if end_frame - 3 == frameIdx:
                    i = i + 1
                    frames = np.asarray(frames)
                    frames = frames.astype(np.uint8)
                    #print(frames.shape)
                    cam_only = video[video.rfind('/', 0, video.rfind('/')) + 1: video.rfind('/')]
                    video_only = video[video.rfind('/') + 1:video.rfind('.')]
                    output_file = os.path.join(out_dir, video_only + '_' + cam_only + '-' + str(i))
                    #print(output_file, label_num(clip_label))
                    skvideo.io.vwrite(output_file + '.avi', frames)
                    fields = [video_index, video[video.rfind('/', 0, video.rfind('/', 0, video.rfind('/')))+1:], init_frame, frameIdx,
                              output_file[output_file.rfind('/', 0, output_file.rfind('/'))+1:] + '.avi', clip_label, label_num(clip_label)]
                    video_index = video_index + 1
                    with open(csv_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)
                    frames = []
