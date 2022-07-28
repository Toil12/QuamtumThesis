import json
import time

class ProjectIO:
    def __init__(self,model_type:str,encod_mode:str,config_name:str):
        self.encode_mode=encod_mode
        self.mode_type=model_type
        self.config_file=open(f"Conifg/setting.json")
        self.parameters= json.load(self.config_file)[config_name]
        self.config_file.close()

        self.name_dict = {
            "normal_a": "Angle",
            "dense_a": "Dense Angle",
            "new":"New"
        }
        self.encode_mode_dict={
            "normal_a":0,
            "dense_a":1,
            "new":2
        }

        time_stamp=time.asctime(time.localtime(time.time()))
        file_tag=f"{model_type}_" \
                 f"{self.encode_mode}_" \
                 f"{self.parameters['n_layers']}_" \
                 f"{self.parameters['re_upload']}_" \
                 f"{time_stamp}"

        self.file_name=f"log_{file_tag}.txt"\
            .replace(" ","_")\
            .replace(":","_")
        self.image_name=f"image_{file_tag}.png"\
            .replace(" ","_")\
            .replace(":","_")
        self.model_name=f"model_{file_tag}.pt"\
            .replace(" ","_")\
            .replace(":","_")

        # operate the files
        open(f"Results/Logs/{self.file_name}", "w").close()

        self.log_file=open(f"Results/Logs/{self.file_name}","a",buffering=1)
        # initialize the log title
        self.write_log(f"start from\n"
                         f"moedl_type:{model_type} at {time_stamp}")
        self.image_title=self.get_image_title()

    def read_parameters(self):
        # data=json.load(self.config_file)[set_name]
        # self.config_file.close()
        self.write_log(f"with encode mode {self.encode_mode}")
        return self.parameters

    def write_log(self,input:str):
        self.log_file.write(input)
        self.log_file.write('\n')

    def get_image_title(self):
        title = "Encoding Breakout Results"
        if self.mode_type == "c":
            m = "Classical "
            title = f"{m} {title}"
        elif self.mode_type == "q":
            m = "Quantum"
            title = f"{m} {self.encode_mode_dict[self.encode_mode]} {title}"
        return title

if __name__ == '__main__':
    # read_parameters()
    oi_o=ProjectIO(model_type="q",
                   encod_mode="dense")
    pa=oi_o.read_parameters()
    print(oi_o.model_name)
    # print(pa["device_name"])
