#mv 20201106_data.txt 20201106_data.txt_back
pmap -x $(pidof mobilenet_light_api) | grep Address >> 20201111_data.txt
while(true)
do
    sleep 20
    pmap -x $(pidof mobilenet_light_api) | grep total >> 20201111_data.txt
done
