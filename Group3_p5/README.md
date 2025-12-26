#Steps to run the code

1. cd Group3_p5/

2. mkdir external

3. Download RAFT model from github: 
   git clone https://github.com/princeton-vl/RAFT.git

4. cd external && ./download_models.sh (This command download the pretrained optical flow model)

5. Add the path of external folder in argparse of main.py line 16

6. Add the sintel model path in the main.py line 405

7. cd Group3_p5/Code/

8. chmod +x main.py

9. python3 main.py



Remember, Everything is given as absolute Path, Have to change it before running main.py, The .pth file of Window Detector is given in U_net model folder