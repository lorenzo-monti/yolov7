#lass Pose

import json 
import ast
from collections import defaultdict
  

class BodyPart:

    def __init__ (self,partId, coordx,coordy): 
        self.PartId = partId
        self.coordx = coordx
        self.coordy = coordy 

    def as_json(self):
        #print(json.dumps(self.__dict__)) 
        return json.dumps(self.__dict__)
    
    def convert_coord_to_pixel(self,heighty, widthx): 
        self.coordx = int(self.coordx * widthx + 0.5)
        self.coordy = int(self.coordy * heighty  + 0.5)
        return self

class Pose: 
    def __init__(self): 
        self.timestamp = 0
        self.body_parts = []
        
    def set_time_stamp(self, timestamp): 
        self.timestamp =timestamp
    def set_body_parts(self, parts): 
        self.body_parts  = parts



    def from_dict(self,dict): 
        self.body_parts = [BodyPart(key,value) for key,value in dict.items()] 

    def from_string_dict(self,string): 
        dict_arr = string.split(';')
        dict_body_parts = [ ast.literal_eval(i) for i in dict_arr] 
        self.body_parts =  [BodyPart(i['BodyPart'],i['Position'][0], i['Position'][1])for i in dict_body_parts]
        
                
   
    def as_json(self,pixel= False, image =False):
        if pixel: 
            heighty,widthx = image.shape[:2]
            body_parts =[part.convert_coord_to_pixel(heighty,widthx) for part in self.body_parts] 
            body_parts =  [BodyPart.as_json(i) for i in body_parts]
        else :body_parts =  [BodyPart.as_json(i) for i in self.body_parts]
        res = defaultdict(list)
        {res[key].append(sub.__dict__[key]) for sub in self.body_parts for key in sub.__dict__} 

        
        dict ={'Timestamp':self.timestamp, 'BodyParts':body_parts}
        return json.dumps(dict)
