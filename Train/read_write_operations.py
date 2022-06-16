import json
import time

class ProjectIO:
    def __init__(self,model_type:str):
        time_stamp=time.asctime(time.localtime(time.time()))
        self.file_name=f"log_{model_type}_{time_stamp}.txt"\
            .replace(" ","_")\
            .replace(":","_")
        self.image_name=f"image_{model_type}_{time_stamp}.png"\
            .replace(" ","_")\
            .replace(":","_")
        self.model_name=f"model_{model_type}_{time_stamp}.pt"\
            .replace(" ","_")\
            .replace(":","_")
        open(f"Results/{self.file_name}", "w").close()
        self.config_file=open("Conifg/setting.json")
        self.log_file=open(f"Results/{self.file_name}","a")
        # initialize the log
        self.write_log(f"start\n"
                         f"moedl_type:{model_type} at {time_stamp}")

    def read_parameters(self,set_name:str='train_test'):
        data=json.load(self.config_file)[set_name]
        self.config_file.close()
        self.write_log(f"with encode mode {data['encode_mode']}")
        return data

    def write_log(self,input:str):
        self.log_file.write(input)
        self.log_file.write('\n')

if __name__ == '__main__':
    # read_parameters()
    oi_o=ProjectIO(model_type="q")
    pa=oi_o.read_parameters()
    print(oi_o.model_name)
    # print(pa["device_name"])
