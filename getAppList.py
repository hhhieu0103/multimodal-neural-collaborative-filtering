import requests, json

URL = "http://api.steampowered.com/ISteamApps/GetAppList/v2"

response = requests.get(URL)
if response.status_code == 200:
    data = response.json()
    with open("AppList.json", "w") as file:
        json.dump(data, file, indent=2)
else:
    print(f'Error: {response.status_code}')