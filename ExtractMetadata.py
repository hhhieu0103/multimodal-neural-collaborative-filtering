#Input Cleaned/Cleaned{ID}.json
#Output Metadata/Metadata{ID}.json
#Start 3900

import json

for file_index in range(0, 5760):
    with open(f'Cleaned/Cleaned{file_index}.json', 'r') as file:
        apps = json.load(file)
        
    for app in apps:
        app.pop('header_image', None)
        app.pop('screenshots', None)
        app.pop('movies', None)
        
    with open(f'Metadata/Metadata{file_index}.json', 'w') as file:
        json.dump(apps, file, indent=2)
        
    print(f'Extracted metadata of file {file_index}')