from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

beg = False
if beg:
    ind = 'tinyimagenet'
    val = 'stl'
    oods_all = ['cifar10', 'cifar100', 'svhn', 'stl', 'tinyimagenet']
    oods_all.remove(ind)
    oods_all.remove(val)
    s="""    Baseline         & 51.41/ 94.24/ 51.16 & 65.46/ 86.66/ 61.69 & 95.7/ 19.5/ 87.24 & 70.86/ 66.8/ 66.7 \\
             Odin             & 53.52/ 92.38/ 51.11 & 72.87/ 75.3/ 66.31 & 98.69/ 6.66/ 91.63 &75.03/ \textbf{58.11}/ 69.68 \\
             Mahalanobis      & 55.31/ 92.2/ 51.56 & 71.69/ 70.15/ 65.81 & 97.18/ 15.68/ 92.92 &74.73/ 59.34/ 70.1 \\
             Self-Supervised  & 58.43/ 92.33/ 55.51 & 75.36/ 86.99/ 70.95 & 92.13/ 58.43/ 91.22 &\textbf{75.31}/ 79.25/ \textbf{72.56} \\
             Generalized-Odin & 65.28/ 87.5/ 60.41 & 59.23/ 91.01/ 56.09 & 34.85/ 99.71/ 35.12 &53.12/ 92.74/ 50.54 \\
             Self-Ensemble    & 54.8/ 97.22/ 55.32 & 52.96/ 93.67/ 52.46 & 91.36/ 33.18/ 68.01 &66.37/ 74.69/ 58.6 \\"""


    # for elem in s.split("&"):
    for elem in s.split('\\'):
        if len(elem) > 2:
            temp = elem.split('&')
            method = temp[0].replace(' ','')
            try:
                auc1, fpr1, acc1 = [float(x) for x in temp[1].replace(' ','').split('/')]
                tnr1 = round(100 - fpr1, 2)
                row_contents1 = [method, ind, oods_all[0], val, auc1, fpr1, tnr1, acc1]
                append_list_as_row('test.csv', row_contents1)
            except:
                print('Failure')
            try:
                auc2, fpr2, acc2 = [float(x) for x in temp[2].replace(' ','').split('/')]
                tnr2 = round(100 - fpr2, 2)
                row_contents2 = [method, ind, oods_all[1], val, auc2, fpr2, tnr2, acc2]
                append_list_as_row('test.csv', row_contents2)
            except:
                print('Failure')
            try:
                auc3, fpr3, acc3 = [float(x) for x in temp[3].replace(' ','').split('/')]
                tnr3 = round(100 - fpr3, 2)
                row_contents3 = [method, ind, oods_all[2], val, auc3, fpr3, tnr3, acc3]
                append_list_as_row('test.csv', row_contents3)
            except:
                print('Failure')
else:
    ind = 'Stanford Dogs'
    val = 'ImageNet-Val'
    oods_all = ['Places', 'Oxford Pets (Common Species)', 'Oxford Pets (Mutually Exclusive)', ('Stanford Dogs SUbsets')]

    s = """Baseline         & 99.69/  1.25/ 97.67                   & 45.82/ 96.02/ 49.57 & 65.55/ 90.66/ 51.23 & 72.54/ 79.57/ 58.53\\
        %  Baseline (MCDO) & 99.7/ 1.1/ 82.13 & 45.84/ 95.75/ 49.98 & 65.65/ 89.99/ 49.98 & \\
         Odin             & \textbf{99.85}/  0.45/ \textbf{98.21} & 45.92/ 96.41/ 49.37 & 62.51/ 93.49/ 50.01 & \textbf{77.51}/ \textbf{76.51}/ 55.50\\
         Mahalanobis      & 99.77/  \textbf{0.30}/ 97.9           & 49.86/ 96.47/ 49.48 & 54.34/ 95.83/ 49.84 & 66.08/ 91.55/ 51.68\\
         Self-Supervised  & 35.28/ 99.25/ 42.07 & 44.58/ 96.68/ 46.33 & 46.27/ 97.08/ 46.65 & 53.20/ 94.21/ 50.40\\
         Generalized-Odin & 96.12/ 22.55/ 88.74         & \textbf{55.89}/ \textbf{94.04}/ \textbf{51.75} & \textbf{65.81}/ \textbf{89.32}/ \textbf{58.02} & 73.10/ 81.10/ \textbf{61.43}\\
         Self-Ensemble & \textbf{99.86}/ 0.35/ 92.07 & 51.86/ 94.88/ 50.53 & 61.85/ 90.16/ 56.34 & - \\"""

    for elem in s.split('\\'):
        if len(elem) > 2:
            temp = elem.split('&')
            method = temp[0].replace(' ','')
            try:
                auc1, fpr1, acc1 = [float(x) for x in temp[1].replace('\textbf{', '').replace('}', '').replace(' ','').split('/')]
                tnr1 = round(100 - fpr1, 2)
                row_contents1 = [method, ind, oods_all[0], val, auc1, fpr1, tnr1, acc1]
                append_list_as_row('fine_grained.csv', row_contents1)
            except:
                print('Failure')
            try:
                auc2, fpr2, acc2 = [float(x) for x in temp[2].replace('\textbf{', '').replace('}', '').replace(' ','').split('/')]
                tnr2 = round(100 - fpr2, 2)
                row_contents2 = [method, ind, oods_all[1], val, auc2, fpr2, tnr2, acc2]
                append_list_as_row('fine_grained.csv', row_contents2)
            except:
                print('Failure')
            try:
                auc3, fpr3, acc3 = [float(x) for x in temp[3].replace('\textbf{', '').replace('}', '').replace(' ','').split('/')]
                tnr3 = round(100 - fpr3, 2)
                row_contents3 = [method, ind, oods_all[2], val, auc3, fpr3, tnr3, acc3]
                append_list_as_row('fine_grained.csv', row_contents3)
            except:
                print('Failure')
            try:
                auc4, fpr4, acc4 = [float(x) for x in temp[4].replace('\textbf{', '').replace('}', '').replace(' ','').split('/')]
                tnr4 = round(100 - fpr4, 2)
                row_contents4 = [method, ind, oods_all[2], val, auc4, fpr4, tnr4, acc4]
                append_list_as_row('fine_grained.csv', row_contents4)
            except:
                print('Failure')