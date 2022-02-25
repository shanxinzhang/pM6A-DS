#!/usr/bin/env python
# _*_coding:utf-8_*_

import re
import itertools

myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
    'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15
}
myTriIndex = {
    'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAU': 3,
    'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACU': 7,
    'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGU': 11,
    'AUA': 12, 'AUC': 13, 'AUG': 14, 'AUU': 15,
    'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAU': 19,
    'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCU': 23,
    'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGU': 27,
    'CUA': 28, 'CUC': 29, 'CUG': 30, 'CUU': 31,
    'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAU': 35,
    'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCU': 39,
    'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGU': 43,
    'GUA': 44, 'GUC': 45, 'GUG': 46, 'GUU': 47,
    'UAA': 48, 'UAC': 49, 'UAG': 50, 'UAU': 51,
    'UCA': 52, 'UCC': 53, 'UCG': 54, 'UCU': 55,
    'UGA': 56, 'UGC': 57, 'UGG': 58, 'UGU': 59,
    'UUA': 60, 'UUC': 61, 'UUG': 62, 'UUU': 63
}

baseSymbol = 'ACGT'


def get_kmer_frequency(sequence, kmer):
    myFrequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        myFrequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
    for key in myFrequency:
        myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
    return myFrequency


def correlationFunction(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
    # 查看index值
    # print('len(myPropertyName)', len(myPropertyName))
    # print('myPropertyName', myPropertyName)
    return CC / len(myPropertyName)


def correlationFunction_type2(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
    return CC


def get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        theta = 0
        for i in range(len(sequence) - tmpLamada - kmer):
            # 查看i的初始
            # print(i)
            theta = theta + correlationFunction(sequence[i:i + kmer],
                                                sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                myPropertyName, myPropertyValue)
        thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        for p in myPropertyName:
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + correlationFunction_type2(sequence[i:i + kmer],
                                                          sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                          myIndex,
                                                          [p], myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def make_PseDNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight):
    encodings = []
    myIndex = myDiIndex
    header = ['#', 'label']
    for pair in sorted(myIndex):
        header.append(pair)
    for k in range(1, lamadaValue + 1):
        header.append('lamada_' + str(k))
    encodings.append(header)
    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        dipeptideFrequency = get_kmer_frequency(sequence, 2)
        thetaArray = get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
        for pair in sorted(myIndex.keys()):
            code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
        for k in range(17, 16 + lamadaValue + 1):
            code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
        encodings.append(code)
    return encodings


