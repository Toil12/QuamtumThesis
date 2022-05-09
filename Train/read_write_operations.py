import json
f=open("../Conifg/setting.json")
def read_parameters():

    data=json.load(f)
    f.close()
    return data
    print(type(data))





if __name__ == '__main__':
    read_parameters()