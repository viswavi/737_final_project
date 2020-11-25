import argparse
import requests
import os
import tarfile


#### just run python prepare-data.py 
### Will create a data_paper diretory with train data for each language pair as well as files for monolingual data
### expects data_paper.tar.gz to be in assign2 folder
def create_data_dir():
    if not os.path.exists("data_paper"):
        data = "data_paper.tar.gz"
        tar = tarfile.open(data, "r:gz")
        tar.extractall()
        tar.close()

def create_fairseq_parallel_data(lang_code, mode, lang_file, en_file):
    with open(lang_file) as lf:
        with open(en_file) as ef:
            with open("ted-"+ mode + ".orig."+lang_code+"-eng","w") as train:
                #Read first file
                lflines = lf.readlines()
                #Read second file
                eflines = ef.readlines()
                #Combine content of both lists  and Write to third file
                for line1, line2 in zip(lflines, eflines):
                    train.write("{} ||| {}\n".format(line1.rstrip(), line2.rstrip()))


if __name__ == "__main__":
    create_data_dir()
    os.chdir("data_paper")

    codes_full = ["tur", "deu", "ron"]
    codes_short = ["tr", "de", "ro"]
    for i in range(3):
        create_fairseq_parallel_data(codes_full[i], "train", "train."+codes_short[i]+"-en."+codes_short[i], "train."+codes_short[i]+"-en.en")
        create_fairseq_parallel_data(codes_full[i], "dev", "dev."+codes_short[i]+"-en."+codes_short[i], "dev."+codes_short[i]+"-en.en")
        create_fairseq_parallel_data(codes_full[i], "test", "test."+codes_short[i]+"-en."+codes_short[i], "test."+codes_short[i]+"-en.en")



    
