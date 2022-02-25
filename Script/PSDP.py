import re,sys
import numpy as np
import pandas as pd

def readRNAFasta(file):
	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input RNA sequence must be fasta format.')
		sys.exit(1)
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGU-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta


def CalculateMatrix(data, order):
    matrix = np.zeros((len(data[0]), 4))
    for i in range(len(data[0])):  # position
        for j in range(len(data)):
            if re.search('-', data[j][i:i+1]):
                pass
            else:
                matrix[i][order[data[j][i:i+1]]] += 1
    return matrix


def PSTNPss(fastas,label):

    for i in fastas:
        if re.search('[^ACGU-]', i[1]):
            print('Error: illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by this encoding scheme.')
            return 0

    encodings = []
    header = ['#']
    for pos in range(len(fastas[0][1])):
        header.append('Pos.%d' %(pos+1))
    encodings.append(header)


    positive = []
    negative = []
    positive_key = []
    negative_key = []
    for i,sequence in enumerate(fastas):
        if int(label[i]) == 1:
           positive.append(sequence[1])
           positive_key.append(sequence[0])
        else:
           negative.append(sequence[1])
           negative_key.append(sequence[0])

    nucleotides = ['A', 'C', 'G', 'U']
    trinucleotides = [n1 for n1 in nucleotides]
    order = {}
    for i in range(len(trinucleotides)):
        order[trinucleotides[i]] = i

    matrix_po = CalculateMatrix(positive, order)
    matrix_ne = CalculateMatrix(negative, order)

    positive_number = len(positive)
    negative_number = len(negative)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for j in range(len(sequence)):
            if re.search('-', sequence[j: j+1]):
               code.append(0)
            else:
                p_num, n_num = positive_number, negative_number
                po_number = matrix_po[j][order[sequence[j: j+1]]]
                ne_number = matrix_ne[j][order[sequence[j: j+1]]]
                code.append(po_number/p_num - ne_number/n_num)
        encodings.append(code)
    return encodings,positive_key, negative_key

# test
input_data=r'sample_datasets.txt'
fastas=readRNAFasta(input_data)
m1=len(fastas)
label1=np.ones((int(m1/2),1))  # Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
vector,A1,A2=PSTNPss(fastas,label)
csv_data=pd.DataFrame(data=vector)
csv_data.to_csv(r'data_PSNP.csv', header=False, index=False)