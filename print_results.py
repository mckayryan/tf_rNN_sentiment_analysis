import json

with open('results.json', 'r') as feedjson:
    result = json.load(feedjson)
print(result)
for i in result:
    print(i)
