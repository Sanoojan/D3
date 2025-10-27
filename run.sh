# python utils/video2frame.py --dataset-path dataset/DeepfakeDatasets/FakeAVCeleb


# python utils/folder2csv.py --is-real True --dataset-path dataset/DeepfakeDatasets/FakeAVCeleb --folders RealVideo-RealAudio
# python utils/folder2csv.py --is-real False --dataset-path dataset/DeepfakeDatasets/FakeAVCeleb --folders FakeVideo-FakeAudio FakeVideo-RealAudio RealVideo-FakeAudio
# python eval.py --gpu-id 0 --loss l2 --encoder XCLIP-16 --real-csv GenVideo/csv/real_MSRVTT.csv --fake-csv GenVideo/csv/Crafter.csv

python eval.py --gpu-id 0 --loss cos --encoder XCLIP-16 --real-csv dataset/DeepfakeDatasets/FakeAVCeleb/csv/RealVideo-RealAudio.csv --fake-csv dataset/DeepfakeDatasets/FakeAVCeleb/csv/FakeVideo-RealAudio.csv