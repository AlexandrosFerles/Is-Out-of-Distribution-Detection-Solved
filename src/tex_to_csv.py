from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

beg = True
if beg:
    ind = 'stl'
    val = 'tinyimagenet'
    oods_all = ['cifar10', 'cifar100', 'svhn', 'stl', 'tinyimagenet']
    oods_all.remove(ind)
    s=""" Baseline & 51.41/ 94.24/ 50.94 & 65.46/ 86.66/ 60.42 & 95.7/ 19.5/ 74.01 & 70.86/ 66.8/ 61.79 \\ 
          Odin              & 53.52/ 92.38/ 52.6 & 72.87/ 75.3/ 66.28  & 98.69/ 6.66/ 82.77 & 75.03/ 58.11/ 67.22 \\
          Mahalanobis       & 55.48/ 92.33/ 53.41 & 70.95/ 71.31/ 65.74 & 96.84/ 18.18/ 86.3 & 74.42/ 60.61/ 68.48 \\
          Self-Supervised   & 58.43/ 92.33/ 56.36 & 75.36/ 86.99/ 70.26 & 92.13/ 58.43/ 81.8 & 75.31/ 79.25/ 69.47 \\
          Generalized-Odin  & 62.97/ 88.81/ 58.5 & 85.02/ 70.78/ 76.89 & 83.95/ 95.47/ 78.69 & 77.31/ 85.02/ 71.36\\
          Self-Ensemble     & 97.08/ 15.09/ 90.57 & 99.38/ 1.95/ 95.4 & 99.99/ 0.0/ 95.69 & 98.82/ 5.68/ 94.2 \\"""
    oods_all.remove(val)


    # for elem in s.split("&"):
    for elem in s.split('\\'):
        if len(elem) > 2:
            temp = elem.split('&')
            method = temp[0].replace(' ','')
            try:
                auc1, fpr1, acc1 = [float(x) for x in temp[1].replace(' ','').split('/')]
                tnr1 = round(100 - fpr1, 2)
                row_contents1 = [method, ind, oods_all[0], val, auc1, fpr1, tnr1, acc1]
                append_list_as_row('standard.csv', row_contents1)
            except:
                print('Failure')
            try:
                auc2, fpr2, acc2 = [float(x) for x in temp[2].replace(' ','').split('/')]
                tnr2 = round(100 - fpr2, 2)
                row_contents2 = [method, ind, oods_all[1], val, auc2, fpr2, tnr2, acc2]
                append_list_as_row('standard.csv', row_contents2)
            except:
                print('Failure')
            try:
                auc3, fpr3, acc3 = [float(x) for x in temp[3].replace(' ','').split('/')]
                tnr3 = round(100 - fpr3, 2)
                row_contents3 = [method, ind, oods_all[2], val, auc3, fpr3, tnr3, acc3]
                append_list_as_row('standard.csv', row_contents3)
            except:
                print('Failure')
else:
    ind = 'ISIC'
    val = 'ImageNet-Val'
    oods_all = ['Places', 'DermoFit (Common Lesion Types)', 'DermoFit (Mutually Exclusive)', 'ISIC Subsets']

    s = """Baseline         & 83.1 / 66.95/ 74.85 & 56.87/92.49/54.49 & 55.91/ 91.18/ 51.59 & 62.34/ 90.10/ 58.26\\
         Odin             & 98.88/ 4.8/ 94.99 & 57.7/ 90.57/ 52.66 & 63.13/ 87.25/ 54.22 & 56.75/ 90.10/ 53.82 \\
         Mahalanobis      & \textbf{99.83}/ \textbf{0.09}/ \textbf{96.37} & \textbf{95.85} / \textbf{25.38}/ \textbf{87.88} & \textbf{91.36}/ \textbf{46.41}/ 76.38 & 60.11/ 92.69/ 51.25 \\
         Self-Supervised  & 73.09/ 80.05/ 66.69 & 67.77/ 85.89/ 63.13 & 80.95/ 75.49/ 73.53 & \textbf{70.07}/ \textbf{83.94}/ \textbf{65}\\
         Generalized-Odin & 85.91/ 48.65/ 77.72 & 69.95/ 69.7/ 65.48  & 65.52/ 76.47/ 59.86 & 43.88/ 93.84/ 48.69\\
         Self-Ensemble    & 99.36/ 2.5/ 93.15 & 74.82/ 81.3/ 62.99 & 87.74/ 56.86/ \textbf{79.65} & -\\"""

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
                print(f'Failure {method}')
            try:
                auc2, fpr2, acc2 = [float(x) for x in temp[2].replace('\textbf{', '').replace('}', '').replace(' ','').split('/')]
                tnr2 = round(100 - fpr2, 2)
                row_contents2 = [method, ind, oods_all[1], val, auc2, fpr2, tnr2, acc2]
                append_list_as_row('fine_grained.csv', row_contents2)
            except:
                print(f'Failure {method}')
            try:
                auc3, fpr3, acc3 = [float(x) for x in temp[3].replace('\textbf{', '').replace('}', '').replace(' ','').split('/')]
                tnr3 = round(100 - fpr3, 2)
                row_contents3 = [method, ind, oods_all[2], val, auc3, fpr3, tnr3, acc3]
                append_list_as_row('fine_grained.csv', row_contents3)
            except:
                print(f'Failure {method}')
            try:
                auc4, fpr4, acc4 = [float(x) for x in temp[4].replace('\textbf{', '').replace('}', '').replace(' ','').split('/')]
                tnr4 = round(100 - fpr4, 2)
                row_contents4 = [method, ind, oods_all[3], val, auc4, fpr4, tnr4, acc4]
                append_list_as_row('fine_grained.csv', row_contents4)
            except:
                print(f'Failure {method}')

