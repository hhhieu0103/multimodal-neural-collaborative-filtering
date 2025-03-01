import json, requests, asyncio, os, csv

LOG_FILE = 'reviews_progress.json'
# Progress: 12      2/34        [XXXXXXXX, YYYYYYYY, ...]
#           file    item/total  cursors

def read_progress():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as file:
            return json.load(file)
    return {
        'file_index': 0,
        'item_index': 0,
        'total_items': None,
        'cursors': []
    }

def write_progress(progress):
    with open(LOG_FILE, 'w') as file:
        json.dump(progress, file, indent=2)
        
def write_reviews_to_csv(app_id, reviews):
    file_path = f'Reviews/Review{app_id}.csv'
    exist = os.path.exists(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['recommendationid', 'author_id', 'review', 'timestamp', 'voted_up', 'weighted_vote_score'])
        if not exist:
            writer.writeheader()
        writer.writerows(reviews)
        
def get_id_list(i):
    with open(f'Cleaned ID/ID{i}.json', 'r') as file:
        return json.load(file)
        
def map_review(raw_review):
    return {
        'recommendationid': raw_review['recommendationid'],
        'author_id': raw_review['author']['steamid'],
        'review': raw_review['review'],
        'timestamp': raw_review['timestamp_updated'],
        'voted_up': raw_review['voted_up'],
        'weighted_vote_score': raw_review['weighted_vote_score'],
    }
        
async def get_review_page(app_id, cursor='*', retries=3, backoff=0.5):
    reviews = []
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "language": "english",
        "num_per_page": 100,
        "cursor": cursor
    }
    
    for attempt in range(1, retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            reviews = data.get('reviews', [])
            break
        elif attempt < retries:
            sleep_time = 0
            if response.status_code == 429:
                sleep_time = 60
                print(f'Too many requests, retry after {sleep_time}')
            else:
                sleep_time = backoff * (2 ** (attempt))
                print(f'Failed to get reviews, retry after {sleep_time}')
            await asyncio.sleep(sleep_time)
            
    return reviews, data.get('cursor', None)
        
async def get_all_reviews(app_id, progress):
    count_page = 0
    while True:
        if len(progress['cursors']) == 0:
            reviews, cursor = await get_review_page(app_id)
        else:
            reviews, cursor = await get_review_page(app_id, progress['cursors'][-1])
        
        if cursor in progress['cursors']:
            write_progress(progress)
            break
        progress['cursors'].append(cursor)
        
        if len(reviews) == 0:
            write_progress(progress)
            break
        
        reviews = list(map(map_review, reviews))
        write_reviews_to_csv(app_id, reviews)
        write_progress(progress)
        count_page += 1
        print(f'Finish getting the {count_page} page of {app_id}')
        await asyncio.sleep(0.5)
        
async def main():
    progress = read_progress()
    
    for file_index in range(progress['file_index'], 5760):
        progress['file_index'] = file_index
        app_ids = get_id_list(file_index)
        progress['total_items'] = len(app_ids)
        
        for app_index in range(progress['item_index'], progress['total_items']):
            app_id = app_ids[app_index]
            progress['item_index'] = app_index
            
            print(f'Begin getting reviews for {app_id}')
            await get_all_reviews(app_id, progress)
            progress['cursors'] = []
            print(f'Finish getting reviews for {app_id}')
            print('=========================================')
            
        progress['item_index'] = 0
            
    # app_id = 1430680
    # progress = {
    #     'file_index': 0,
    #     'item_index': 0,
    #     'total_items': None,
    #     'first_cursor': None,
    #     'next_cursor': None
    # }
    # print(f'Begin getting reviews of {app_id}')
    # await get_all_reviews(app_id, progress)
    # print(f'Finish getting reviews for {app_id}')
    # print('=========================================')
        
if __name__ == '__main__':
    asyncio.run(main())