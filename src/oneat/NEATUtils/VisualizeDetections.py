from pathlib import Path
import json

class VizDet:
    
    def __init__(self, imagefile: Path, jsonfile: Path, csvfile: Path):
        
        self.imagefile = imagefile 
        self.jsonfile = jsonfile 
        self.csvfile = csvfile
        
        
    def load_json(self):
        with open(self.jsonfile) as f:
            return json.load(f)  
        
    def __call__(self):
        
        self.key_catagories = self.load_json()      
        