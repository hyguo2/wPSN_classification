import os
import argparse
import statistics
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-fl', '--file', type = str, help = 'specify the dataset identifier file name', default = '') #To be determined here.

args = parser.parse_args()

id_file = args.file

expdir_result = "./Results/BuildNet/"
if not os.path.exists(expdir_result):
    os.makedirs(expdir_result)

#file writer
#so.popen("module load gcc/7.1.0\n")
#generate raw wPSN files
os.system("chmod u+x ./scripts/towtgdv_all.sh\n")
if 'cath'.upper() in id_file.upper():
    PC = 'CATH'
elif 'scop'.upper() in id_file.upper():
    PC = 'SCOP'
os.system("sh ./scripts/towtgdv_all.sh ./raw_network_files/{0:4s}-all/networks-6-A ./identifiers/{1:s}.txt 2 ./Results/BuildNet/{1:s} -m 4\n".format(PC.lower(), id_file))

#generate CVM matrices
os.system("Rscript --vanilla ./scripts/CVM_calc.R {0:s}".format(id_file))

#run classifier, 5 fold cross validation
expdir_result = "./Results/DNN/" + id_file
if not os.path.exists(expdir_result):
    os.makedirs(expdir_result)

for i in range(5):
    os.system("python3 ./scripts/DNN_classifier.py -ep 500 -l 3 -nh 256 -lr 0.0001 -c {0:s} -gpu 1 -gcn 3 -tl DNN_CV -k {1:d} -cn '[48,96]' ".format(id_file, i))

#output result
rslt_list = os.listdir("./Results/DNN/" + id_file)
rslt = [float(x.split('_')[4]) for x in rslt_list]
np.savetxt('./'+args.file+'_5f_CV_Accuracy_'+str(statistics.mean(rslt))+'.txt',rslt_list,fmt='%0.4f')