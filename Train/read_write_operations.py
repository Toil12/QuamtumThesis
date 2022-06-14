import json

class ProjectIO:
    def __init__(self):
        # open("Results/logs.txt", "w").close()
        self.config_file=open("Conifg/setting.json")
        self.log_file=open("Results/logs.txt","a")

    def read_parameters(self,set_name:str='train'):
        data=json.load(self.config_file)
        self.config_file.close()
        return data[set_name]

    def write_log(self,input:str):
        self.log_file.write(input)
        self.log_file.write('\n')

if __name__ == '__main__':
    # read_parameters()
    oi_o=ProjectIO()
    for _ in range(10):
        oi_o.write_log("bad")