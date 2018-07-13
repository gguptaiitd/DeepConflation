
from Bio import SeqIO
import tempfile
import os
import math
import random


base2num = dict(zip("ACGT", (1, 2, 3, 4)))
num2base = dict(zip((1, 2, 3, 4), "ACGT"))
folder = tempfile.mkdtemp()
image_save = os.path.join(folder, 'refSet')

image_size = 30
numImages = 10000
mutImages = 1
mutLength = int(image_size*0.3)

def makeRefStreamString(ref_fasta_fn):
    images = []
    genes =[]
    image_fname = "dataset/stringdata_"+str(image_size)+"_"+str(mutLength)+".txt"
    print(image_fname)

    image_file = open(image_fname, "w")
 
    for x in range(numImages):
        loc = random.randint(0, len(ref_seq) - (image_size + 1))

        l_seq = image_size
        seq = list(ref_seq[loc:loc + l_seq])

        image_1d = ''.join(seq) 

        actual = image_1d

        s_seed = [random.randint(0, l_seq - mutLength) for _ in range( mutImages)]
        s_rlen = [random.randint(1, mutLength) for _ in range(len(s_seed))]
        for start, l in zip(s_seed, s_rlen):
            s_seq = seq[:]
            subs = []
            [subs.append(random.choice('ACGT')) for _ in range(l)]
            s_seq[start:start + l] = subs
            image_1d = ''.join(s_seq) 

            d=actual+"\t"+image_1d
            print >> image_file,d


#provide name of reference fasta file
file = open("/home/genome/som_data/sequence.fasta","r")

for seq_record in SeqIO.parse(file, "fasta"):
    ref_seq = str(seq_record.seq)

makeRefStreamString(ref_seq)
