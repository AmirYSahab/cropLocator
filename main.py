'''
Created on Nov 18, 2020

@author: Amir Yeganehsahab
'''
## ____________________________________________________________________________________
__author__= ['Amir Yeganehsahab']
__url__ = (' ')
__version__= '0.0'
__doc__= '''
            reads images and prints out the 4 corner coordinates of the croped region
            inputs: two images (one is a rotated/nonrotated crop of the base image) 
            output: coordinates of the most ptobable croped region
            '''
__inputs__ = '''
            base_image_path: str
            croped_image_path: str
            '''
__outputs__ = '''
            coordinates of the most probable croped region 
            including: 
                upper left corner (ulc): tuple
                lower right corner (lrc): tuple
            '''
## ____________________________________________________________________________________           
import cv2, os, argparse, multiprocessing
from multiprocessing import Pool, cpu_count
import numpy as np
## ____________________________________________________________________________________

class __embed__():
    def __init__(self,StarMap = None,crop = None):
        self.pathes = {'StarMap':StarMap,'crop':crop}
        
        if StarMap == None or crop == None:
            self.pathes['StarMap'] = '{}/inputs/StarMap.png'.format(os.path.dirname(os.path.realpath(__file__)))
            self.pathes['crop'] = '{}/inputs/Small_area.png'.format(os.path.dirname(os.path.realpath(__file__)))
        
        self.arrays = {'StarMap':list,\
                       'crop':list}
## ____________________________________________________________________________________        
class __app__(argparse.Action):
    def __init__(self,args):
        
        self.data = __embed__()
        self.data.pathes['StarMap'] = args.StarMap[0]
        self.data.pathes['crop'] = args.crop[0]
        print(self.data.pathes['crop'])
        self.counter = 0
        self.__read__(self.data, args.resize)
        
        self.crop = self.data.arrays['crop']
        self.base_image = self.data.arrays['StarMap']
        self.h_c,self.w_c, self.c_c = np.shape(self.crop)
        self.h_b,self.w_b, self.c_b = np.shape(self.base_image)
        
        self.rotations = ['0','cv2.ROTATE_90_CLOCKWISE','cv2.ROTATE_90_COUNTERCLOCKWISE','cv2.ROTATE_180']
        self.angles = [0,90,-90,180]
        
        with Pool(cpu_count()) as p:
            outputs = p.map(self.repeat_for_rotation,self.rotations)
        p.close()
        arg_ = np.argmax([o for i,(c,o) in enumerate(outputs)])
        proper_center = outputs[arg_][0]

        self.ulc =(int(proper_center[0]-self.h_c/2),int(proper_center[1]-self.w_c/2))
        self.urc =(int(proper_center[0]+self.h_c/2),int(proper_center[1]-self.w_c/2))
        self.lrc =(int(proper_center[0]+self.h_c/2),int(proper_center[1]+self.w_c/2))
        self.llc =(int(proper_center[0]-self.h_c/2),int(proper_center[1]+self.w_c/2))
        
        self.image =cv2.rectangle(self.base_image,self.ulc,self.lrc,(255,255,255),1)
        
        print('upper left corner: ',self.ulc,'\n',\
              'upper right corner: ', self.urc,'\n',\
              'lowe left corner: ', self.llc,'\n',\
              'lower right corner: ',self.lrc)

        cv2.imshow('*', self.image)
        cv2.waitKey()
        
    def __read__(self, data, resize):
        
        for key,path in data.pathes.items():
            extension = path.split(os.sep)[-1].split('.')[-1]
            
            if  extension == 'png' or extension == 'jpg':
                image = cv2.imread(path)
                if resize == False or resize == None:
                    data.arrays[key] = image
                else:
                    r = eval(resize)
                    data.arrays[key] = cv2.resize(image,(int(np.shape(image)[1]*r),int(np.shape(image)[0]*r)))
                 
    def similarity(self, crop):
        similarity_mat = np.zeros((self.h_b-self.h_c,self.w_b-self.w_c))
        similarity_mat = self.fetch(similarity_mat,crop)
        ind = np.unravel_index(np.argmax(similarity_mat, axis=None), similarity_mat.shape)  # returns a tuple
        return ind, np.max(similarity_mat)
    
    def repeat_for_rotation(self, angle):
        print('doing it for {}'.format(angle))
        self.counter += 0
        crop = cv2.rotate(self.crop, eval(angle))
        return self.similarity(crop)
        
    def fetch(self,input,crop):            
        for l,r in enumerate(input):
            for p,c in enumerate(r):
                crop_ = np.array(self.base_image[l:l+self.h_c,p:p+self.w_c])
                '''
                    calculate cross correlation of two matrices 
                    and extract the correlation value as a measure of similarity
                '''
                input[l,p] = np.corrcoef(crop.ravel(),crop_.ravel())[1,0]
        return input
## ____________________________________________________________________________________       
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find the crop region, example run: python3 main.py --StarMap=~/inputs/StarMap.png --crop=~/inputs/Small_area.png, --resize=1/5')
    
    parser.add_argument('--StarMap', metavar='-S', type=str, nargs=1,
                            help='path to StarMap image')
    
    parser.add_argument('--crop', metavar='-C', type=str, nargs=1,
                            help='path to croped image')
    
    parser.add_argument('--resize', metavar='-r', type=str,
                            help='if you are in rush resize the image by a fraction e.q. 1/5 etc. \n if you do not define it it will not resize the images')
    args = parser.parse_args()

    app = __app__(args)