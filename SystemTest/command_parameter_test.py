import sys
import getopt

model_type = None
config_name = None
encode_mode = None

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "m:c:e:",["model=","config=","encode="])  # 短选项模式

except:
    print("Error")

print(opts)
for opt, arg in opts:
    if opt in ['--model']:
        model_type = arg
    elif opt in ['--config']:
        config_name = arg
    elif opt in ['--encode']:
        encode_mode = arg

print(model_type)
print(config_name)
print(encode_mode)
