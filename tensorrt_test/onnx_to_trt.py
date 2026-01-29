import tensorrt as trt

# 初始化 TensorRT
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 配置优化器配置
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 设置最大工作空间大小，例如 1GB

# 解析 ONNX 文件
parser = trt.OnnxParser(network, logger)
with open("policy.onnx", "rb") as model:
    parser.parse(model.read())

# 构建 TensorRT 引擎
engine = builder.build_engine(network, config)

# 保存 TensorRT 引擎文件
with open("policy.trt", "wb") as f:
    f.write(engine.serialize())
