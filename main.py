from Train.read_write_operations import ProjectIO

oi_o=ProjectIO(model_type="q",
               encod_mode="dense")
pa=oi_o.read_parameters("train_lightning")
print(pa['device_name'])