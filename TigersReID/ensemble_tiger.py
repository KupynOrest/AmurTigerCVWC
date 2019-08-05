import csv
import pandas as pd # not key to functionality of kernel

sub_files = ['submissions_wild/submission_wild_1.json',
             'submissions_wild/submission_wild_2.json',
             'submissions_wild/submission_wild_3.json'
]

# Weights of the individual subs
sub_weight = [0.7268302465204549**2,
              0.7491973824230916**2,
              0.7538995084828835**2,
              ]

abc = pd.read_csv(sub_files[0])
xyz = pd.read_csv(sub_files[1])

print(abc.head())
print(xyz.head())

Hlabel = 'Image'
Htarget = 'Id'
npt = 5  # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = (1 / (i + 1))

print(place_weights)

lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    ## input files ##
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    with open(file) as json_file:
        data = json.load(json_file)
#         reader = csv.DictReader(open(file, "r"))
        sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

print()
## output file ##
out = open("sub_ens_29submits.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel, Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt, 0) + (place_weights[ind] * sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()