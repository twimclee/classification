import torch
import onnx
from efficientnet_pytorch import EfficientNet

model_name = f'efficientnet-b{6}'
model = EfficientNet.from_name(model_name, num_classes=2)

model.load_state_dict(torch.load(r"D:\models\noul\add\20230322\noul_cont_68.pt"))
model.eval()
device = torch.device("cuda:0")
model = model.to(device)

# model = EfficientNet.from_pretrained('efficientnet-b1')
dummy_input = torch.randn(1, 3, 528, 528).to(device)
# batch, channel, size, size

dummy_output = model(dummy_input)


SAVE = r"D:\models\noul\add\20230322\noul_cont_68.onnx"

model.set_swish(memory_efficient=False)
torch.onnx.export(model, 
	dummy_input, 
	SAVE, 
	# opset_version=12,
	training=torch.onnx.TrainingMode.EVAL,
	do_constant_folding=True,
	export_params=True,
	verbose=False,)


model = onnx.load(SAVE)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)