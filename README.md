# Preprocessing

First clone the repository. <br><br>
Download the following files and put in repository under the folder name pretrained_models
<ul>
  <li>[Model weights](https://drive.google.com/file/d/1JDBgiwEFpBHMtIJLRd1y_9IRKmw99MgN/view)
  <li>[Dlib CNN face detector](https://drive.google.com/file/d/1l2R9qImsBXkCgk698v1QUp0k7rqudeRd/view)
  <li>[Dlib shape predictor](https://drive.google.com/file/d/1bLLe01Bw8SNdVZIJjBTKriqFNHZlaPQL/view)
</ul>
Next dowload FFMPeg from this link https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z .<br><br>
Extract the downloaded .7z file to C:/ drive and add the bin path inside the extracted folder to environment variables.<br><br>
Now open CMD(put cmd in windows search it will come) and go to the repository folder.<br><br>
# In cmd run(just copy paste in sequence)

```bash
conda create -n pre pip python==3.9 -y
```
```bash
conda activate pre
```
```bash
pip install -r requirements.txt
```
```bash
conda create -n pre pip python==3.9 -y
```
```bash
python preprocesing.py --base_path data --crf 42
```
After running these commands in cmd u can see the cropped images of face and keyframes of the same inside the data/original/failure_project and data/compressed_42/failure_project/crops <b><b>(just open the data folder and view images in sub folders you will get idea of what happened).
