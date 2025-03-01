import json, os, logging

LOG_FILE = 'clean.log'
MAX_NUMBER_OF_FILES = 5760
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def filter_app(app):
    id = next(iter(app))
    if app[id]['success']:
        data = app[id]['data']
        is_game = data['type'] == 'game'
        is_released = data['release_date']['coming_soon'] == False
        for_window = data['platforms']['windows']
        return is_game and is_released and for_window
    return False

def map_app(app):
    id = next(iter(app))
    app_data = app[id]['data']
    fields_to_delete = ['type', 'required_age', 'capsule_image', 'capsule_imagev5', 'website', 'packages', 'package_groups', 'support_info', 'background', 'background_raw', 'content_descriptors', 'ratings', 'platforms', 'mac_requirements', 'linux_requirements', 'dlc', 'legal_notice', 'achievements', 'ext_user_account_notice', 'reviews', 'controller_support', 'demos', 'metacritic', 'drm_notice' ]
    
    for field in fields_to_delete:
        app_data.pop(field, None)
    
    app_data['release_date'] = app_data['release_date']['date']
    
    categories = list(map(lambda category: str(category['description']), app_data.get('categories', [])))
    app_data['categories'] = ", ".join(categories)
    
    genres = list(map(lambda genre: str(genre['description']), app_data.get('genres', [])))
    app_data['genres'] = ", ".join(genres)
    
    developers = list(map(lambda developer: developer, app_data.get('developers', [])))
    app_data['developers'] = ", ".join(developers)
    
    publishers = list(map(lambda publisher: publisher, app_data.get('publishers', [])))
    app_data['publishers'] = ", ".join(publishers)
    
    pc_requirement = app_data.get('pc_requirements', None)
    if pc_requirement:
        app_data['min_requirements'] = pc_requirement.get('minimum', None)
        app_data['rec_requirements'] = pc_requirement.get('recommended', None)
    app_data.pop('pc_requirements', None)
    
    app_data['total_recommendations'] = app_data.get('recommendations', {}).get('total', 0)
    app_data.pop('recommendations', None)
    
    app_data['price'] = app_data.get('price_overview', {}).get('final', None)
    app_data.pop('price_overview', None)
    
    return app_data

def read_progress():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as file:
            lines = file.readlines()
            if lines:
                return int(lines[-1].strip().split(": ")[1])
                
    return None

def main():
    next_file = read_progress()
    
    if next_file:
        next_file += 1
    else:
        next_file = 0
    
    for i in range(0, 5760):
        
        with open(f'Detail Lists/Detail{i}.json') as file:
            apps = json.load(file)
            
        apps = list(filter(filter_app, apps))
        apps = list(map(map_app, apps))
        
        with open(f'Cleaned/Cleaned{i}.json', 'w') as file:
            json.dump(apps, file, indent=2)

        print(f'Cleaned Detail{i}')
        logging.info(f"Last processed: {i}")
        
        # print(apps[0]['1430720']['data'].keys())
        # print(map_app(apps[0]['1430720']['data']))
        
if __name__ == '__main__':
    main()