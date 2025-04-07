import json

with open("test.json") as f:
    d = json.load(f)

print(d)
print(d["phase0"]["sim"][0])
print(d["phase0"]["train"][0])
print(d["phase0"]["selection"][0])
print(d["phase0"]["agent"][0])
print(d["phase1"]["sim"][0])
print(d["phase1"]["train"][0])
print(d["phase1"]["selection"][0])
print(d["phase1"]["agent"][0])

print(d["phase0"]["sim"][1])
print(d["phase0"]["train"][1])
print(d["phase0"]["selection"][1])
print(d["phase0"]["agent"][1])
print(d["phase1"]["sim"][1])
print(d["phase1"]["train"][1])
print(d["phase1"]["selection"][1])
print(d["phase1"]["agent"][1])

