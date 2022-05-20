import json
f=open("../Conifg/setting.json")
def read_parameters(set_name:str='train'):
    data=json.load(f)
    f.close()
    return data[set_name]

if __name__ == '__main__':
    read_parameters()