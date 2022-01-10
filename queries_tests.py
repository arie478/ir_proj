import json
import requests
from time import time

with open('queries_train.json', 'rt') as f:
    queries = json.load(f)


# Average precision calculation function
def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


# place the domain you got from ngrok or GCP IP below.
url = "http://35.194.18.78:8080"

# Send the test qeuries to the server and get the results
qs_res = []
for q, true_wids in queries.items():
    duration, ap = None, None
    t_start = time()
    try:
        res = requests.get(url + '/search', {'query': q}, timeout=35)
        duration = time() - t_start
        print(res.status_code)
        if res.status_code == 200:
            pred_wids, _ = zip(*res.json())
            ap = average_precision(true_wids, pred_wids)
    except Exception as exc:
        pass

    qs_res.append((q, duration, ap))

# Calculate the results for the MAP@40 test
i = 0
sumAp = 0
sumTime = 0
success = 0
fail = 0
failed = []
for title_time_ap40 in qs_res:
    if title_time_ap40[2] is not None:
        sumAp += title_time_ap40[2]
        success += 1
    else:
        fail += 1
        failed.append(title_time_ap40[0])
    if title_time_ap40[1] is None:
        sumTime += 0
    else:
        sumTime += title_time_ap40[1]
    i += 1


# Print results of the MAP@40 test in readable format :
print("TRAINING SET RESULTS")
print("-------------------")
print("Average time per query :")
print(sumTime / i)
print("Average Mean Precision for K = 40 : (MAP@40)")
print(sumAp / i)
print("Succeeded querying :")
print(success)
print("Failed querying :")
print(fail)
if fail > 0:
    print(failed)
print("-------------------")
