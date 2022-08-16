from .seven_scenes import SevenScenes
from .seven_scenes import SevenScenesVal
from .cambridge_landmarks import Cambridge
#from .cambridge_landmarks import Cambridge_val
from .twelve_scenes import TwelveScenes


def get_dataset(name):

    return {
            '7S' : SevenScenes,
            '7S_val' : SevenScenesVal,
            
            'Cambridge' : Cambridge,
            #'Cambridge_val' : Cambridge_val,
            
            '12S' : TwelveScenes,
            # '12S_val' : TwelveScenes_val,    
           }[name]

