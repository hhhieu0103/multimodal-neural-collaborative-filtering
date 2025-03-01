import os, json, asyncio, aiohttp, aiofiles, random

LOG_FILE = 'multimedia_progress.json'
CONCURRENT_LIMIT = 10
MAX_RETRIES = 3
RETRY_DELAY = 2

class TooManyRequestsError(Exception):
    def __init__(self, message):
        super().__init__(message)

def read_progress():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as file:
            return json.load(file)
    return {
        'file_index': 0,
    }

def write_progress(file_index):
    progress = {
        'file_index': file_index,
    }
    with open(LOG_FILE, 'w') as file:
        json.dump(progress, file, indent=2)
        
def create_directory(batch_index, root_dir='E:/Multimedia'):
    path = os.path.join(root_dir, batch_index) 
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f'Directory for batch {batch_index} created')
        
    return path

async def download_file(sem, session, url, filename):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with sem:
                async with session.get(url) as response:
                    if response.status == 200:
                        async with aiofiles.open(filename, 'wb') as f:
                            await f.write(await response.read())
                        print(f"Downloaded {filename}")
                        return
                    elif response.status == 429:
                        raise TooManyRequestsError("Received HTTP 429: Too Many Requests.")
                    else:
                        print(f"Failed to download {filename}: {response.status}")
                        return
        except (aiohttp.ClientError, asyncio.TimeoutError, TooManyRequestsError) as e:
            retries += 1
            print(f"Error downloading {filename}. Retry {retries}/{MAX_RETRIES}...")
            if retries < MAX_RETRIES:
                if isinstance(e, TooManyRequestsError):
                    delay = 60
                else:
                    delay = RETRY_DELAY * (2 ** retries) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                print(f"Failed to download {filename} after {MAX_RETRIES} retries.")

def get_download_urls(app, batch_index):
    id = str(app['steam_appid'])
    root_dir = create_directory(batch_index)
    urls = []
    urls.append((app['header_image'], os.path.join(root_dir, f'{id}-header.jpg')))
            
    movies = app.get('movies', [])
    if len(movies) > 0:
        file_path = os.path.join(root_dir, f'{id}-main.mp4')
        urls.append((movies[0]['mp4']['max'], file_path))
        
    return urls

async def main():
    progress = read_progress()
    
    for file_index in range(progress['file_index'], 5760):
        with open(f'Cleaned/Cleaned{file_index}.json') as file:
            apps = json.load(file)
            
        urls = []
        for app in apps:
            urls.extend(get_download_urls(app, str(file_index)))
            
        sem = asyncio.Semaphore(CONCURRENT_LIMIT)
        async with aiohttp.ClientSession() as session:
            tasks = [download_file(sem, session, url, file_path) for url, file_path in urls]
            await asyncio.gather(*tasks)
            
        write_progress(file_index + 1)
            
        # for item_index in range(progress['item_index'], len(apps)):
        #     app = apps[item_index]
        #     id = str(app['steam_appid'])
            
        #     app_dir = create_directory(id)
            
        #     # header_dir = create_directory(id, 'Header Image')
        #     # screenshots_dir = create_directory(id, 'Screenshots')
        #     # movies_dir = create_directory(id, 'Movies')
            
        #     urls = []
        #     urls.append((app['header_image'], os.path.join(app_dir, 'header.jpg')))
            
        #     movies = app.get('movies', [])
        #     if len(movies) > 0:
        #         file_path = os.path.join(app_dir, str(movies[0]['id'])) + '.mp4'
        #         urls.append((movies[0]['mp4']['max'], file_path))
            
        #     # for movie in app.get('movies', []):
        #     #     file_path = os.path.join(app_dir, str(movie['id'])) + '.mp4'
        #     #     urls.append((movie['mp4']['max'], file_path))
            
        #     # for screenshot in app.get('screenshots', []):
        #     #     file_path = os.path.join(screenshots_dir, str(screenshot['id'])) + '.jpg'
        #     #     urls.append((screenshot['path_full'], file_path))
                
        #     sem = asyncio.Semaphore(CONCURRENT_LIMIT)
        #     async with aiohttp.ClientSession() as session:
        #         tasks = [download_file(sem, session, url, file_path) for url, file_path in urls]
        #         await asyncio.gather(*tasks)
        #         write_progress(file_index, item_index)
        #         print(f'Finish item {item_index}/{len(apps)}')
                
        print(f'Finish batch {file_index}')
 
        await asyncio.sleep(1)
            
if __name__ == '__main__':
    asyncio.run(main())
