import json, os

LOG_FILE = 'cleaned_id.txt'

def write_progress(index):
    with open(LOG_FILE, 'w') as file:
        file.write(str(index))
        
def read_progress():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as file:
            return int(file.readline())
    return None

def main():
    start = read_progress()
    if start is not None:
        start += 1
    else:
        start = 0
        
    for i in range(start, 5760):
        with open(f'Cleaned/Cleaned{i}.json', 'r') as file:
            apps = json.load(file)
            
        ids = list(map(lambda app: app['steam_appid'], apps))
        
        with open(f'Cleaned ID/ID{i}.json', 'w') as file:
            json.dump(ids, file, indent=2) 
        
        write_progress(i)
        print(f'Finish file {i}')
    
if __name__ == '__main__':
    main()