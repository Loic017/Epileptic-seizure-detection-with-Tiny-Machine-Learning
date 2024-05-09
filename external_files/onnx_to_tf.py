import onnx
from onnx_tf.backend import prepare

seizure_type = "x"
model_type = "x"

path = f"x"
onnx_model = onnx.load(path)  # load onnx model
tf_rep = prepare(onnx_model)
tf_rep.export_graph(f"x")
print("COMPLETED CONVERSION")
