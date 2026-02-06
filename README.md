通过yolov5模型训练遥感遥感风力发电机遥感检测数据  建立深度学习 风力基于YOLO的遥感风力发电机检测系统
@[toc]

遥感风力发电机遥感检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/92a39086c6444c26bd5ce98002a55521.png)

1268张1类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/890f88cc0ff243848dce407e6b2dee07.png)



数据集划分：训练集：验证集=80%：20%，具体如下：
总图片数量：1268 张;
训练集数量：1014 张;
验证集数量：254 张;![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5bd60b89450843a89d6ea22c1bbbf909.png)


data.yaml文件;names: ['windTurbine'] names: [风力发电机']

['windTurbine']: 图片数量：'1268'，标注框数：'4452'

基于YOLO的模型用于遥感风力发电机检测，我们需要按照以下步骤进行操作。这里假设的是Ultralytics的YOLOv5或类似版本作为模型基础。

### 1. 数据集准备

首先确保你的数据集已经根据给定的比例（80%训练集，20%验证集）划分好，并且所有图像和对应的标注文件（YOLO格式）都已准备好。每个图像应该有一个对应的`.txt`文件，包含该图像中的对象标注信息。

### 2. 创建 `data.yaml` 文件

创建一个名为 `data.yaml` 的配置文件来定义数据集路径、类别信息等。根据提供的信息，这个文件应如下所示：

```yaml
train: ./images/train # 训练集图片目录
val: ./images/val # 验证集图片目录

nc: 1 # 类别数量
names: ['windTurbine'] # 类别名称
```

注意：在你的描述中，`names` 字段出现了两次，一次是英文名，另一次是中文名。在这里我们使用英文名，如果你需要支持多语言，请考虑在应用层处理翻译问题。

### 3. 安装依赖

确保安装了必要的库。如果还没有安装，可以通过以下命令安装：

```bash
pip install ultralytics
```

### 4. 开始训练

使用 Ultralytics YOLO 提供的脚本来开始训练过程。以下是基本命令示例：

```bash
yolo task=detect mode=train model=yolov5s.pt data=path/to/data.yaml epochs=100 imgsz=640 batch=16
```

- `task=detect`: 指定任务为检测。
- `mode=train`: 模式为训练。
- `model=yolov5s.pt`: 使用YOLOv5的小型预训练模型作为起点。如果硬件资源允许，可以尝试更大规模的模型如`yolov5m.pt`, `yolov5l.pt` 或 `yolov5x.pt`。
- `data=path/to/data.yaml`: 指向之前创建的 `data.yaml` 文件。
- `epochs=100`: 设置训练周期数。
- `imgsz=640`: 设置输入图像尺寸。
- `batch=16`: 设置批次大小。请根据你的GPU内存调整此值。

### 5. 监控与评估

训练过程中，YOLO会自动保存最佳权重到指定的运行目录下（默认为 `runs/detect/train/weights/best.pt`）。你可以通过TensorBoard或其他方式监控训练进度。

训练完成后，可以在验证集上评估模型性能：

```bash
yolo task=detect mode=val model=path/to/best.pt data=path/to/data.yaml
```

这将输出包括精确度、召回率、mAP等在内的评价指标。

### 6. 测试与部署

使用训练好的模型对新的图像进行预测：

```bash
yolo task=detect mode=predict model=path/to/best.pt source=path/to/test/images
```

这会在指定的输出目录生成带有边界框标记的预测结果图像。

为了构建一个完整的基于YOLO的遥感风力发电机检测系统，我们将按照以下步骤详细说明整个流程：数据集准备与预处理、划分数据集、创建`data.yaml`文件、环境搭建、模型训练、推理与部署、性能评估以及模型导出。以下是每个步骤的详细指南和代码示例。

### 1. 数据集准备与预处理

确保你的数据集已经准备好，并且图像和标注（YOLO格式）一一对应。假设你的数据集结构如下：

```
windTurbineDataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

#### 标注文件转换
如果你的标注不是YOLO格式，需要转换。例如，使用Python脚本将其他格式转换为YOLO格式：

```python
import os

def convert_to_yolo_format(input_file, output_dir, image_width, image_height):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    yolo_lines = []
    for line in lines:
        parts = line.strip().split()
        # 假设输入格式为 [class_id, x_min, y_min, width, height]
        class_id, x_min, y_min, obj_width, obj_height = map(float, parts)
        
        # 转换为中心点坐标及归一化
        x_center = (x_min + obj_width / 2) / image_width
        y_center = (y_min + obj_height / 2) / image_height
        width = obj_width / image_width
        height = obj_height / image_height
        
        yolo_lines.append(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
    
    output_path = os.path.join(output_dir, os.path.basename(input_file))
    with open(output_path, 'w') as file:
        file.writelines(yolo_lines)

# 示例调用
convert_to_yolo_format('path/to/your/input.txt', 'path/to/output/dir', 1920, 1080)
```

### 2. 划分数据集

根据提供的信息，总共有1268张图片，其中80%作为训练集，20%作为验证集。可以手动或编写脚本进行划分：

```python
import os
import random
import shutil

source_images_dir = 'windTurbineDataset/images'
source_labels_dir = 'windTurbineDataset/labels'

train_images_dir = 'windTurbineDataset/images/train'
val_images_dir = 'windTurbineDataset/images/val'
train_labels_dir = 'windTurbineDataset/labels/train'
val_labels_dir = 'windTurbineDataset/labels/val'

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

image_files = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg')]
random.shuffle(image_files)

split_point = int(len(image_files) * 0.8)

for i, image_file in enumerate(image_files):
    label_file = image_file.replace('.jpg', '.txt')
    
    if i < split_point:
        shutil.move(os.path.join(source_images_dir, image_file), os.path.join(train_images_dir, image_file))
        shutil.move(os.path.join(source_labels_dir, label_file), os.path.join(train_labels_dir, label_file))
    else:
        shutil.move(os.path.join(source_images_dir, image_file), os.path.join(val_images_dir, image_file))
        shutil.move(os.path.join(source_labels_dir, label_file), os.path.join(val_labels_dir, label_file))
```

### 3. 创建 `data.yaml` 文件

在项目根目录下创建 `data.yaml` 文件：

```yaml
train: ./windTurbineDataset/images/train
val: ./windTurbineDataset/images/val

nc: 1
names: ['windTurbine']
```

### 4. 环境搭建

安装必要的依赖：

```bash
pip install ultralytics
```

### 5. 模型训练

开始训练模型：

```bash
yolo task=detect mode=train model=yolov5s.pt data=path/to/data.yaml epochs=100 imgsz=640 batch=16
```

### 6. 推理与部署

使用训练好的模型对新的图像进行预测：

```bash
yolo task=detect mode=predict model=path/to/best.pt source=path/to/test/images
```

### 7. 性能评估

在验证集上评估模型性能：

```bash
yolo task=detect mode=val model=path/to/best.pt data=path/to/data.yaml
```

### 8. 模型导出

导出模型以供后续部署使用（如ONNX格式）：

```bash
yolo export model=path/to/best.pt format=onnx
```

### 完整代码示例

这里提供了一个简化的完整代码示例，涵盖了从数据准备到模型导出的主要步骤：

```python
# 数据集划分代码
import os
import random
import shutil

def split_dataset(source_images_dir, source_labels_dir, train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)
    split_point = int(len(image_files) * 0.8)

    for i, image_file in enumerate(image_files):
        label_file = image_file.replace('.jpg', '.txt')
        
        if i < split_point:
            shutil.move(os.path.join(source_images_dir, image_file), os.path.join(train_images_dir, image_file))
            shutil.move(os.path.join(source_labels_dir, label_file), os.path.join(train_labels_dir, label_file))
        else:
            shutil.move(os.path.join(source_images_dir, image_file), os.path.join(val_images_dir, image_file))
            shutil.move(os.path.join(source_labels_dir, label_file), os.path.join(val_labels_dir, label_file))

# 调用函数
split_dataset(
    'windTurbineDataset/images',
    'windTurbineDataset/labels',
    'windTurbineDataset/images/train',
    'windTurbineDataset/images/val',
    'windTurbineDataset/labels/train',
    'windTurbineDataset/labels/val'
)

# 训练模型命令
# yolo task=detect mode=train model=yolov5s.pt data=path/to/data.yaml epochs=100 imgsz=640 batch=16

# 推理命令
# yolo task=detect mode=predict model=path/to/best.pt source=path/to/test/images

# 性能评估命令
# yolo task=detect mode=val model=path/to/best.pt data=path/to/data.yaml

# 导出模型命令
# yolo export model=path/to/best.pt format=onnx
```

基于YOLO的遥感风力发电机检测系统的开发。这个流程包括了从数据准备、模型训练、性能评估到最后的模型导出。

以上文字及代码仅供参考学习。
