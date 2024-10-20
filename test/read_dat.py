import pickle
from pathlib import Path


#f=open(r"F:/DeepModels/LiaeUDT512_SAEHD/LiaeUDT512_SAEHD_data.dat","rb")
#model_data = pickle.load(f)
model_data_path=Path(r"F:/DeepModels/LiaeUDT512_SAEHD/LiaeUDT512_SAEHD_data.dat")
model_data = pickle.loads (model_data_path.read_bytes() )

print(model_data.get('iter',0))
print(model_data.get('resolution',"没有resolution"))
options=model_data.get("options",None)
print(options)
print(options.get('resolution',0))