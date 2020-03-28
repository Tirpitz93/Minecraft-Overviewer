cd "C:/Datafile/LSelter/Documents/Minecraft-Overviewer/test_conf"
$dirname =  "C:/Datafile/LSelter/Documents/Minecraft-Overviewer/test_conf"
$timestamp = $(((get-date).ToUniversalTime()).ToString("yyyy-MM-dd-hh-mm-ss"))
echo $timestamp
echo "Clearing output Directory"
rm  "C:/Datafile/LSelter/Documents/Minecraft-Overviewer/test_conf/output/*" -r -Force

py "C:/Datafile/LSelter/Documents/Minecraft-Overviewer/overviewer.py" -c"C:/Datafile/LSelter/Documents/Minecraft-Overviewer/test_conf/conf.py" --forcerender --force-render -p6 |
        Tee-Object -FilePath "$dirname/logs/$timestamp.log"
echo "Starting Server"
Start-Process "http://localhost:8001"