
<div align="center">

<h1>飞鱼去字幕: 去除视频中字幕的工具</h1>

</div>
飞鱼去字幕是一个视频字幕擦除工具，内部集成了VSR开源模型,STTN模型，LAMA模型，PROPAINTER模型，满足大家日常工作研究的需求。
大家如果喜欢这个项目就star一下咯。

---


## 注意事项
- 需要N卡，最低1060上面运行
- 50系显卡，需要手动调整一下requirements.txt的依赖在按照
- 代码在python 3.12 ，conda12 上面验证通过


## 功能点
- 支持批量视频处理
- 支持VSR，STTN，LAMA，PROPAINTER模型
- 方便的UI操作界面

## 源码运行

#### 安装 

1. Clone Repo

   ```bash
   git clone https://github.com/ppzflpp/videosubtitleremover.git
   ```

2. 创建环境（默认你已经按照了python3.12版本）

   ```bash
   # 创建虚拟环境
   python -m venv myenv
   myenv\Scripts\activate
   # 安装依赖
   pip install -r requirements.txt 
   ```

#### 准备模型
权重文件需要放到 `./models` 下面对应文件夹.  
权重下载路径 [夸克](https://pan.quark.cn/s/4e17998328ea) 


The directory structure will be arranged as:
```
models:
├─big-lama
│   big-lama.pt
├─propainter
│   ProPainter.pth
│   raft-things.pth
│   recurrent_flow_completion.pth
└─sttn
    infer_model.pth
    sttn.pth
```

#### 开始运行
```shell
cd videosubtitleremover
python main.py 
```
接下来会启动UI页面，按照UI页面操作即可。

#### 自行打包exe文件
```shell
#运行时会有黑色的console弹出，方便定位问题（调试推荐）
pyinstaller   --add-data="models/sttn/*;models/sttn" --add-data="models/big-lama/*;models/big-lama" --add-data="models/propainter/*;models/propainter" --add-data="tools/ffmpeg/bin/*;tools/ffmpeg/bin/" main.py
#运行时没有任何弹出（正式版推荐）
pyinstaller --windowed  --add-data="models/sttn/*;models/sttn" --add-data="models/big-lama/*;models/big-lama" --add-data="models/propainter/*;models/propainter" --add-data="tools/ffmpeg/bin/*;tools/ffmpeg/bin/" main.py

```


## EXE运行
下载地址：
链接：https://pan.quark.cn/s/ad102839e525


## 欢迎一起讨论交流学习
QQ群：  1036028422


## 训练


