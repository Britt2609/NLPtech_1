import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import pandas as pd



# Read in files
def read_in():
    with open('vaccination_stance_annotations.csv', encoding="utf8") as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        line_count = 0
        annotation = []
        for row in csv_read:
            if line_count == 0:
                print(f'Column names are{", ".join(row)}')
                line_count += 1
            else:
                annotation += row[3]
                line_count += 1
        print(annotation)
        ids = [i for i in range(25)]
        print(ids)

        with open('student2.csv', encoding="utf8") as csv_file:
            csv_read2 = csv.reader(csv_file, delimiter=',')
            line_count2 = 0
            annotation2 = []
            for row in csv_read2:
                if line_count2 == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count2 += 1
                else:
                    annotation2 += row[3]
                    line_count2 += 1
            print(annotation2)

        score = cohen_kappa_score(annotation2, annotation)
        print(score)

        labels = np.unique(annotation2)
        a = confusion_matrix(annotation2, annotation, labels=labels)

        conf_mat = pd.DataFrame(a, index=labels, columns=labels)
        print(conf_mat)


read_in()
