import numpy as np
import pandas as pd
import os

def read(file_name:str):
    doc=[]
    with open(f"{file_name}", "r") as f:
        for line in f:
            doc.append([line])
    doc=doc[3:]
    return doc

def reorder(document:[str]):
    head=[]
    data=[]
    for line in range(len(document)):
        line_split = document[line][0].strip().split(',')
        pd_line = []
        for item in line_split:
            context=item.split(':')

            if line==0:
                h=context[0]
                head.append(h)

            pd_line.append(context[1])
        data.append(pd_line)

    data_frame=pd.DataFrame(data,columns=head)
    return data_frame


def run():
    doc_name1 = "log_q_dense_a_3_False_Mon_Jul_11_18_48_00_2022.txt"
    doc_name2="log_q_dense_a_5_False_Mon_Jul_11_18_48_01_2022.txt"
    doc_name3="log_q_dense_a_3_False_Wed_Jul_20_18_24_48_2022.txt"

    docs=[doc_name1,doc_name2,doc_name3]
    for d in docs:
        doc=read(d)
        # print(doc)
        data_frame=reorder(doc)
        time=data_frame['time consume']
        time=list(map(float,time))
        print(f"mean episode time is {np.mean(time)}")

if __name__ == '__main__':
    run()