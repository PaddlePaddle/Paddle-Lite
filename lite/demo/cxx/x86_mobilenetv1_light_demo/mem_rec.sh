#mv 20201106_data.txt 20201106_data.txt_back
pmap -x 23868 | grep Address >> 20201110_data.txt
while(true)
do
    sleep 50
    pmap -x 23868 | grep total >> 20201110_data.txt
done
