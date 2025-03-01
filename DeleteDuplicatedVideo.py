import os, glob

dup_video_name = '*-main.mp4'

for i in range(0, 1000):
    dup_videos = glob.glob(os.path.join(f'E:/Multimedia/{i}', dup_video_name))
    if i == 13:
        continue
    for video in dup_videos:
        os.remove(video)
        print(f'Deleted {video}')