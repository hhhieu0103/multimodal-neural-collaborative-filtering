import json, os, glob, ffmpeg, time

LOG_FILE = 'compress-progress.json'
VIDEOS_PATH = 'E:/Multimedia'

#Progress: batch

def write_log(progress):
    with open(LOG_FILE, 'w') as file:
        json.dump(progress, file)
        
def read_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as file:
            return json.load(file)
    return {
        'batch': 0
    }
    
def get_videos_list(batch):
    all_videos = glob.glob(os.path.join(VIDEOS_PATH, str(batch), '*.mp4'))
    uncompressed_videos = [video for video in all_videos if '-compressed' not in video]
    return uncompressed_videos

def get_video_output_path(video_path):
    name_with_ext = os.path.basename(video_path)
    name_without_ext = os.path.splitext(name_with_ext)[0]
    dir_name = os.path.dirname(video_path)
    output_name = name_without_ext + '-compressed' + '.mp4'
    output_path = os.path.join(dir_name, output_name)
    return output_path

# def get_video_resolution(video_path):
#     try:
#         probe = ffmpeg.probe(video_path)
#         video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
#         width = int(video_stream['width'])
#         height = int(video_stream['height'])
#         return width, height
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
    
def get_video_info(input_video):
    probe = ffmpeg.probe(input_video, v='error', select_streams='v:0', show_entries='stream=codec_name,width,height')
    
    codec = probe['streams'][0].get('codec_name', None)
    width = probe['streams'][0].get('width', None)
    height = probe['streams'][0].get('height', None)

    return codec, width, height

def compress_video(video_path):
    output_path = get_video_output_path(video_path)
    if os.path.exists(output_path):
        print(f'{video_path} is already compressed')
        return None, None
    
    original_size = bytes_to_megabytes(os.path.getsize(video_path))
    if original_size < 20:
        print(f'File is smaller than 20 MB, skipping ...')
        return None, None
    
    start = time.time()
    ffmpeg_cmd = (
        ffmpeg
        .input(video_path, hwaccel="cuda")
    )

    codec, width, height = get_video_info(video_path)
    if width > 1920 or height > 1080:
        print(f"Scaling {video_path} to 1080p while keeping aspect ratio...")
        ffmpeg_cmd = ffmpeg_cmd.filter("scale", "if(gt(iw/ih,1920/1080),1920,-2)", "if(gt(ih/iw,1080/1920),1080,-2)")
    else:
        print(f"Skipping scaling for {video_path} (already 1080p or lower).")
        
    if codec == 'hevc':
        print(f'The video already encoded with HEVC, skipping ...')
        return None, None
        
    print(f'Compressing {video_path} ({original_size} MB) ...')
    ffmpeg_cmd = (
        ffmpeg_cmd
        .output(output_path, vcodec="hevc_nvenc", preset="slow", acodec="copy")
        .run(overwrite_output=True, quiet=True)
    )
    
    print(f"Finished compressing: {output_path}")
    time_taken = round(time.time() - start,2)
    compressed_size = bytes_to_megabytes(os.path.getsize(output_path))
    print(f"Deleting original file: {video_path}")
    os.remove(video_path)
    
    print(f'File size reduced from {original_size} MB to {compressed_size} MB')
    print(f'Saved {round(original_size - compressed_size,2)} MB ({cal_reduced_percentage(compressed_size, original_size)}%)')
    print(f'Took {time_taken} seconds to finish')
    print('====================================================================================')
    return output_path, time_taken
    
def get_total_size(file_list):
    total_size = 0
    for file_path in file_list:
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    return bytes_to_megabytes(total_size)

def bytes_to_megabytes(bytes):
    return round(bytes/1024/1024, 2)

def cal_reduced_percentage(compressed, uncompressed):
    return round((1 - (compressed/uncompressed))*100, 2)

def seconds_to_minutes(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return minutes, remaining_seconds
       
def main():
    last_batch = read_log()['batch']
    
    for batch in range(last_batch + 1,5760):
        total_time = 0
        uncompressed_videos = get_videos_list(batch)
        if len(uncompressed_videos) == 0:
            print('No videos to compress')
            continue
        compressed_videos = []
        uncompressed_size = get_total_size(uncompressed_videos)
        for i, video in enumerate(uncompressed_videos):
            print(f'Processed {i}/{len(uncompressed_videos)}')
            compressed_video, time_taken = compress_video(video)
            if compress_video != None and time_taken != None:
                compressed_videos.append(compressed_video)
                total_time += time_taken
        write_log({'batch': batch})
        print(f'Finish batch {batch}')
        compressed_size = get_total_size(compressed_videos)
        reduced_percentage = cal_reduced_percentage(compressed_size, uncompressed_size)
        print(f'Batch size reduced from {uncompressed_size} MB to {compressed_size} MB')
        print(f'Saved {round(uncompressed_size - compressed_size, 2)} MB ({reduced_percentage}%)')
        minutes, seconds = seconds_to_minutes(total_time)
        print(f'Took {minutes} minutes and {round(seconds,2)} seconds to finish')
    
if __name__ == '__main__':
    main()