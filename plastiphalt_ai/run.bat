@echo off
python D:\plastiphalt_ai_project\plastiphalt_ai\app.py video_pipeline ^
  --video D:\plastiphalt_ai_project\plastiphalt_ai\dataset\waste_conveyor2.mp4 ^
  --model D:\plastiphalt_ai_project\plastiphalt_ai\models\plastic_rf.joblib ^
  --frame_skip 30 ^
  --severity high ^
  --traffic 25000 ^
  --temperature 35 ^
  --available "{\"HDPE\":300,\"PP\":200,\"LDPE\":100,\"PET\":50}" ^
  --bitumen 1000 ^
  --agg_ratio 10
pause
