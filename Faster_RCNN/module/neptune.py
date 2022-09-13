import subprocess
import sys

try:
    import neptune.new as neptune
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neptune-client"])
    import neptune.new as neptune
    

def neptune_log(token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YTJmMGZiOC1jYzc0LTRkNTYtYWU1YS1jMGI0YmNmZDU4ZjgifQ=="):
    run = neptune.init(
        project="sungsu/Faster-R-CNN",
        api_token=token,
    ) 
    return run

def record_image(run,img):
    run["outputs/rpn_valid"].log(neptune.types.File.as_image(img))