export LD_LIBRARY_PATH=${PWD}:${LD_LIBRARY_PATH}

./ocr_db_crnn \
    ocr_img_model/ch_det_mv3_db_opt.nb \
    ocr_img_model/ch_rec_mv3_crnn_opt.nb \
    ocr_img_model/ppocr_keys_v1.txt \
    ocr_img_model/1.jpg \
    ./
