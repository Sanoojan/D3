# python utils/video2frame.py --dataset-path dataset/DeepfakeDatasets/DFDC

# python utils/folder2csv.py --is-real False --dataset-path GenVideo --folders Crafter Gen2 HotShot Lavie ModelScope MoonValley MorphStudio Show_1 Sora WildScrape

# python utils/folder2csv.py --is-real True --dataset-path dataset/DeepfakeDatasets/FakeAVCeleb --folders RealVideo-RealAudio RealVideo-FakeAudio
# python utils/folder2csv.py --is-real False --dataset-path dataset/DeepfakeDatasets/FakeAVCeleb --folders FakeVideo-FakeAudio FakeVideo-RealAudio 


# python eval.py --gpu-id 0 --loss l2 --encoder XCLIP-16 --real-csv GenVideo/csv/real_MSRVTT.csv --fake-csv GenVideo/csv/Crafter.csv

# python eval.py --gpu-id 0 --loss cos --encoder XCLIP-16 --real-csv dataset/DeepfakeDatasets/FakeAVCeleb/csv/RealVideo-RealAudio.csv --fake-csv dataset/DeepfakeDatasets/FakeAVCeleb/csv/FakeVideo-RealAudio.csv

# python eval_retina.py --gpu-id 1 --loss cos --encoder XCLIP-16 --real-csv GenVideo/csv/real_MSRVTT.csv --fake-csv GenVideo/csv/Crafter.csv
# python eval.py --gpu-id 0 --loss l2 --encoder XCLIP-16 --real-csv GenVideo/csv/real_MSRVTT.csv --fake-csv GenVideo/csv/Crafter.csv




# python utils/video2frame.py --dataset-path dataset/DeepfakeDatasets/DFDC/video/test
python utils/folder2csv.py --is-real True --dataset-path dataset/DeepfakeDatasets/DFDC --folders test/real_videos
python utils/folder2csv.py --is-real False --dataset-path dataset/DeepfakeDatasets/DFDC --folders test/fake_videos