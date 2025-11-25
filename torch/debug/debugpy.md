# How to debug Pytorch code in vscode

1. 进入 vscode 的 debug 选项卡（找不到图标则通过快捷键 CTRL+SHIFT+D），点击 `创建 launch.json` 文件

2. 在弹窗中选择 `Python Debugger -> Python File`

3. 此时代码编辑区会弹出 `launch.json` 文件，在右下角还有一个 `Add configuration`

4. 编辑 `launch.json` 文件，program 改为所需 debug 的代码的文件名，`justMyCode` 设为 `false` 使得 debug 程序可以进到非本项目的代码中

5. 保存编辑好的 `launch.json` 文件，并在debug选项卡中选择刚才新增的配置