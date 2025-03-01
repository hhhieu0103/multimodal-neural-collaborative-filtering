import json, os, asyncio, aiohttp, sys

URL = "https://store.steampowered.com/api/appdetails"

def get_number_of_lists(dir='ID Lists'):
    return len([file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))])

def get_id_list(i):
    with open(f'ID Lists/GameIDList{i}.json', 'r') as file:
        id_list = json.load(file)
    return id_list

async def get_detail(session, id, retries=3, backoff=0.5):
    params = {"appids": id, "cc": "us", "l": "en"}
    for attempt in range(retries):
        try: 
            async with session.get(URL, params=params) as response:
                response.raise_for_status()
                detail = await response.json()
                if detail and detail.get(str(id), None).get('success', None):
                    return id, 'Completed', detail
                else:
                    return id, 'Removed', None
        except aiohttp.ClientError as e:
            if attempt == retries - 1:
                return id, 'Error', str(e)
            elif e.status == 429:
                print('Too many requests')
                await asyncio.sleep(120)
            else:
                sleep_time = backoff * (2 ** (attempt + 1))
                print(f"Error fetching {id}: {e}. Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
        
def write_to_file(buffer, file_name='Detail Lists/Detail.json'):
    with open(file_name, 'w') as file:
        json.dump(buffer, file, indent=2)
    
    
async def main():
    number_of_lists = get_number_of_lists()

    async with aiohttp.ClientSession() as session:
        for i in range(5169, number_of_lists):
            id_list = get_id_list(i)
            detail_buffer = []
            failed_ids = []
            
            for count, id in enumerate(id_list):
                id, status, detail = await get_detail(session, id)
                if status == 'Completed':
                    detail_buffer.append(detail)
                elif status == 'Error':
                    failed_ids.append(id)
                print(f'Finish detail {count} of list {i}')
                await asyncio.sleep(1.25)
            
            write_to_file(detail_buffer, f'Detail Lists/Detail{i}.json')
            if len(failed_ids) > 0:
                write_to_file(failed_ids, f'Failed Lists/Failed{i}.json') 
            print(f'Success: {len(detail_buffer)}\nError: {len(failed_ids)}')

if __name__ == '__main__':
    asyncio.run(main())