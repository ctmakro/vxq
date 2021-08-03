## 倒茶机配置方法

1. 在桌上机器人臂展可以够到的范围内，放置4个marker，编号为0到3，组成一个正方形，这个正方形的尺寸应该尽量大。

2. 运行 `python experiment_table_config_generator.py`，根据提示填入三个测量参数，生成新的`table_config.json`文件。

3. 运行 `python experiment_table_config_verify.py <model>`，其中`<model>`可以为：

    - release: 普通版
    - debug/original: 开发版（臂较长）
    - shorter: 短版（狗版）
    - tea: 倒茶机专版

      请根据臂长配置选择合适的<model>；具体臂长设置参见`vxq_hid.py`

    机械臂会依次移动到marker 0-3的上方并停留，等待用户按下回车继续。请根据机械臂到达的实际位置，微调marker的摆放位置，确保每个marker都落在机械臂末端的正下方。

4. 运行 `python vxq_tea_attendant_demo.py <model>` 检验效果。在摄像头的辅助下，机械臂应能精确抵达marker 5所在位置。

5. 倒茶上位机地址 http://127.0.0.1:9001/
