from csv import writer
import numpy as np
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

beg = True
if beg:
    # ind = 'stl'
    # val = 'tinyimagenet'
    # oods_all = ['cifar10', 'cifar100', 'svhn', 'stl', 'tinyimagenet']
    # oods_all.remove(ind)
    s="""
         Baseline          & 80.23/ 75.22/ 73.19 & 93.13/ 39.89/ 83.15 & 75.05/ 80.97/ 68.26 & 82.8/ 65.36/ 74.87 \\
         Odin             & 83.75/ 70.36/ 76.43 & 97.52/ 14.77/ 86.97 & 79.38/ 74.64/ 71.93 & 86.88/ 53.26/ 78.44 \\
         Mahalanobis      & 93.8/ 45.61/ 88.0 & 98.43/ 2.73/ 94.12 & 83.52/ 81.03/ 66.99 & 91.92/ 43.12/ 83.04 \\
         Self-Supervised  & 96.31/ 13.11/ 91.07 & 96.84/ 17.87/ 87.78 & 99.71/ 1.2/ 97.31 & \\
         Generalized-Odin & 79.93/ 79.78/ 73.61 & 76.82/ 97.93/ 73.36 & 77.71/ 74.48/ 70.74 & 78.15/ 84.06/ 72.57 \\
         Self-Ensemble    & 87.36/ 52.45/ 79.01 & 99.61/ 2.15/ 89.71 & 75.82/ 82.68/ 65.85 & 87.6/ 45.76/ 78.19 \\
    """
    # oods_all.remove(val)

    # for elem in s.split('\\'):
    #     if len(elem) > 2:
    #         temp = elem.split('&')
    #         method = temp[0].replace(' ','')
    #         try:
    #             auc1, fpr1, acc1 = [float(x) for x in temp[1].replace(' ','').split('/')]
    #             tnr1 = round(100 - fpr1, 2)
    #             row_contents1 = [method, ind, oods_all[0], val, auc1, fpr1, tnr1, acc1]
    #             append_list_as_row('standard.csv', row_contents1)
    #         except:
    #             print('Failure')
    #         try:
    #             auc2, fpr2, acc2 = [float(x) for x in temp[2].replace(' ','').split('/')]
    #             tnr2 = round(100 - fpr2, 2)
    #             row_contents2 = [method, ind, oods_all[1], val, auc2, fpr2, tnr2, acc2]
    #             append_list_as_row('standard.csv', row_contents2)
    #         except:
    #             print('Failure')
    #         try:
    #             auc3, fpr3, acc3 = [float(x) for x in temp[3].replace(' ','').split('/')]
    #             tnr3 = round(100 - fpr3, 2)
    #             row_contents3 = [method, ind, oods_all[2], val, auc3, fpr3, tnr3, acc3]
    #             append_list_as_row('standard.csv', row_contents3)
    #         except:
    #             print('Failure')
    ret = ""
    for elem in s.split('\\'):
        if len(elem) > 2:
            temp = elem.split('&')
            method = temp[0].replace(' ','')
            # val_set = temp[0].replace(' ','')
            ret +=f"{method} & "
            try:
                auc1, fpr1, acc1 = [float(x) for x in temp[1].replace(' ','').split('/')]
                tnr1 = round(100 - fpr1, 2)
                ret += f"{auc1}/ {tnr1}/ {acc1} & "
            except:
                print('Failure')
            try:
                auc2, fpr2, acc2 = [float(x) for x in temp[2].replace(' ','').split('/')]
                tnr2 = round(100 - fpr2, 2)
                ret += f"{auc2}/ {tnr2}/ {acc2} & "
            except:
                print('Failure')
            try:
                auc3, fpr3, acc3 = [float(x) for x in temp[3].replace(' ','').split('/')]
                tnr3 = round(100 - fpr3, 2)
                ret += f"{auc3}/ {fpr3}/ {acc3} & "
            except:
                print('Failure')
            mean_auc = round(np.mean([auc1, acc2, auc3]), 2)
            mean_tnr = round(np.mean([tnr1, tnr2, tnr3]), 2)
            mean_acc = round(np.mean([acc1, acc2, acc3]), 2)
            ret += f"{mean_auc}/ {mean_tnr}/ {mean_acc} \\\\"
            # ret+="\n"
    print(ret)

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

