@echo off
REM AutoDL 云平台数据上传辅助脚本（在本地Windows运行）
REM 用于打包项目和数据，方便上传到云平台

echo ========================================
echo  云平台数据打包工具
echo ========================================
echo.

REM 设置变量
set PROJECT_NAME=Graduation_Project_CHD-MedIA
set OUTPUT_DIR=E:\graduation\cloud_upload

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo [1/3] 打包项目代码...
echo 排除: node_modules, .git, __pycache__, output_slices
powershell -Command "& {Compress-Archive -Path 'E:\graduation\%PROJECT_NAME%\backend', 'E:\graduation\%PROJECT_NAME%\frontend', 'E:\graduation\%PROJECT_NAME%\*.md', 'E:\graduation\%PROJECT_NAME%\*.txt', 'E:\graduation\%PROJECT_NAME%\*.sh', 'E:\graduation\%PROJECT_NAME%\*.bat' -DestinationPath '%OUTPUT_DIR%\project_code.zip' -Force}"

echo.
echo [2/3] 打包训练数据...
echo 请手动压缩以下目录:
echo   E:\BaiduNetdiskDownload\mr_train\
echo 压缩为: %OUTPUT_DIR%\mmwhs_data.zip
echo.
pause

echo.
echo [3/3] 生成上传清单...
(
echo ========================================
echo  云平台上传文件清单
echo ========================================
echo.
echo 1. 项目代码:
echo    文件: project_code.zip
echo    大小: 
powershell -Command "& {(Get-Item '%OUTPUT_DIR%\project_code.zip').Length / 1MB | ForEach-Object {'{0:N2} MB' -f $_}}"
echo.
echo 2. 训练数据:
echo    文件: mmwhs_data.zip
echo    说明: 需要手动压缩
echo.
echo 3. 上传到AutoDL:
echo    - 在AutoDL网页打开JupyterLab
echo    - 点击上传按钮
echo    - 上传到 /root/autodl-tmp/
echo.
echo 4. 解压命令（在AutoDL终端执行）:
echo    unzip /root/autodl-tmp/project_code.zip -d /root/
echo    unzip /root/autodl-tmp/mmwhs_data.zip -d /root/autodl-tmp/
echo.
echo 5. 运行配置脚本:
echo    cd /root/%PROJECT_NAME%
echo    bash backend/training/setup_cloud.sh
echo.
echo ========================================
) > "%OUTPUT_DIR%\upload_instructions.txt"

echo.
echo ========================================
echo  打包完成！
echo ========================================
echo.
echo 输出目录: %OUTPUT_DIR%
echo.
echo 文件列表:
dir "%OUTPUT_DIR%" /b
echo.
echo 请查看: %OUTPUT_DIR%\upload_instructions.txt
echo.
start explorer "%OUTPUT_DIR%"
pause
