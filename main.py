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
import os, argparse, multiprocessing
from multiprocessing import Pool, cpu_count

try:
    import cv2
except:
    command = 'sudo apt-get install python3-dev python3-numpy'
    os.system(command)
    import cv2

try:
    import numpy as np
except:
    command = 'python3 -m pip install -U numpy'
    os.system(command)
    import numpy as np
    
try:
    from skimage.measure import structural_similarity as ssim
except:
    try:
        command = 'python3 -m pip install -U scikit-image'
        os.system(command)
        from skimage.measure import structural_similarity as ssim
    except:
        from skimage.measure import compare_ssim as ssim
## ____________________________________________________________________________________

class __embed__():
    def __init__(self,StarMap = None,crop = None, resize = '1/5'):
        self.pathes = {'StarMap':StarMap,'crop':crop}
        self.resize = resize
        if StarMap == None or crop == None:
            self.pathes['StarMap'] = '{}/inputs/StarMap.png'.format(os.path.dirname(os.path.realpath(__file__)))
            self.pathes['crop'] = '{}/inputs/Small_area.png'.format(os.path.dirname(os.path.realpath(__file__)))
        
        self.arrays = {'StarMap':list,\
                       'crop':list}
## ____________________________________________________________________________________        
class __app__(argparse.Action):
    def __init__(self,args=None):
        try:
            self.type == args.type
            if args.type == None:
                self.type = 'ssim'
        except:
            self.type = 'ssim'
            
        self.counter = 0
        self.data = __embed__()
        try:
            self.data.pathes['StarMap'] = args.StarMap[0]
            self.data.pathes['crop'] = args.crop[0]
            print(self.data.pathes['crop'])
            self.__read__(self.data, args.resize)
        except:
            self.__read__(self.data, self.data.resize)
            pass
            
        
        self.crop = self.data.arrays['crop']
        
        cv2.imshow('crop',self.crop)
        
        self.base_image = self.data.arrays['StarMap']
        
        cv2.imshow('StarMap',self.base_image)
        cv2.waitKey()
        
        self.r_c,self.c_c = np.shape(self.crop)
        self.r_b,self.c_b = np.shape(self.base_image)
        print('self.crop',np.shape(self.crop))
        print('self.base_image',np.shape(self.base_image))
        '''
        self.rotations = ['0','cv2.ROTATE_90_CLOCKWISE','cv2.ROTATE_90_COUNTERCLOCKWISE','cv2.ROTATE_180']
        self.angles = [0,90,-90,180]
        '''
        self.rotations = self.angles = list(np.arange(-90,90))
        with Pool(cpu_count()) as p:
            outputs = p.map(self.repeat_for_rotation,self.rotations)
        p.close()
        arg_ = np.argmax([o for i,(c,o) in enumerate(outputs)])
        # coordinates in similarity matrix
        proper_center = outputs[arg_][0]
        # coordinates in original image
        proper_center = (proper_center[0]+self.r_c/2,proper_center[1]+self.c_c/2)
        # coorner coordinates
        self.ulc =(int(proper_center[0]-self.r_c/2),int(proper_center[1]-self.c_c/2))
        self.llc =(int(proper_center[0]+self.r_c/2),int(proper_center[1]-self.c_c/2))
        self.lrc =(int(proper_center[0]+self.r_c/2),int(proper_center[1]+self.c_c/2))
        self.urc =(int(proper_center[0]-self.r_c/2),int(proper_center[1]+self.c_c/2))
        crop = self.base_image[self.ulc[0]:self.lrc[0],self.ulc[1]:self.lrc[1]]
        cv2.imshow('crop',crop)
        self.image =cv2.rectangle(self.base_image,self.ulc,self.lrc,(255,255,255),2)
        
        logs = {'upper left corner': self.ulc,\
              'upper right corner': self.urc,\
              'lowe left corner': self.llc,\
              'lower right corner': self.lrc}
        
        print('____________________________________________________')
        print('the crop corners are as below:')
        print(logs)
        print('____________________________________________________')
        output_path = '{}/outputs'.format(os.path.dirname(os.path.realpath(__file__)))
        
        self.check_path(output_path)
        imagefile = '{}/output.png'.format(output_path)
        logfile = '{}/corners.txt'.format(output_path)
        # save image
        cv2.imwrite(imagefile,self.image)
        print('____________________________________________________')
        print('you can see the crop location on original image in:')
        print(imagefile)
        print('____________________________________________________')
        # write outputs
        with open(logfile,'w') as log:
            log.write(str(logs))
        print('____________________________________________________')
        print('you can get the resulting corner coordinates in text file below:')
        print(logfile)
        print('____________________________________________________')
        
        cv2.imshow('*', self.image)
        cv2.waitKey()
        
    def __read__(self, data, resize):
        
        for key,path in data.pathes.items():
            extension = path.split(os.sep)[-1].split('.')[-1]
            
            if  extension == 'png' or extension == 'jpg':
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if resize == False or resize == None:
                    data.arrays[key] = image
                else:
                    r = eval(resize)
                    data.arrays[key] = cv2.resize(image,(int(np.shape(image)[1]*r),int(np.shape(image)[0]*r)))
    
    def check_path(self,path):
        if not os.path.isdir(path):
            print('path \n {} \n is created'.format(path))
            os.mkdir(path)
            
    def similarity(self, crop):
        similarity_mat = np.zeros((self.r_b-self.r_c,self.c_b-self.c_c))
        if self.type == 'corrcoef':
            similarity_mat = self.fetch_corrcoef(similarity_mat,crop)
        elif self.type == 'mse':
            similarity_mat = self.fetch_mse(similarity_mat,crop)
        elif self.type == 'ssim':
            similarity_mat = self.fetch_ssim(similarity_mat,crop)
        ind = np.unravel_index(np.argmax(similarity_mat, axis=None), similarity_mat.shape)  # returns a tuple
        return ind, np.max(similarity_mat)
    
    def repeat_for_rotation(self, angle):
        print('doing it for {}'.format(angle))
        self.counter += 0
        #crop = cv2.rotate(self.crop, eval(angle))
        crop = self.rotate_image(self.crop, angle)
        return self.similarity(crop)
        
    def fetch_corrcoef(self,input,crop):            
        for l,r in enumerate(input):
            for p,c in enumerate(r):
                crop_ = np.array(self.base_image[l:l+self.r_c,p:p+self.c_c])
                '''
                    calculate cross correlation of two matrices 
                    and extract the correlation value as a measure of similarity
                '''
                input[l,p] = np.corrcoef(crop.ravel(),crop_.ravel())[1,0]
        return input
    
    def fetch_mse(self,input,crop):            
        for l,r in enumerate(input):
            for p,c in enumerate(r):
                crop_ = np.array(self.base_image[l:l+self.r_c,p:p+self.c_c])
                '''
                    calculate cross correlation of two matrices 
                    and extract the correlation value as a measure of similarity
                '''
                input[l,p] = self.__mse__(crop_, crop)
        return input
    
    def fetch_ssim(self,input,crop):            
        for l,r in enumerate(input):
            for p,c in enumerate(r):
                crop_ = np.array(self.base_image[l:l+self.r_c,p:p+self.c_c])
                '''
                    calculate cross correlation of two matrices 
                    and extract the correlation value as a measure of similarity
                '''
                input[l,p] = ssim(crop_, crop)
        return input
    
    def __mse__(self, a, b):
        # source: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        mse = np.sum((a.astype("float") - b.astype("float")) ** 2)
        mse /= float(a.shape[0] * b.shape[1])
        
        return mse
    
    def rotate_image(self,image, angle):
        # this keeps aspect ration
        # source : https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
## ____________________________________________________________________________________       
   
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='find the crop region, example run: python3 main.py --StarMap=~/inputs/StarMap.png --crop=~/inputs/Small_area.png, --resize=1/5')
    
    parser.add_argument('--StarMap', metavar='-S', type=str, nargs=1,
                            help='path to StarMap image')
    
    parser.add_argument('--crop', metavar='-C', type=str, nargs=1,
                            help='path to croped image')
    
    parser.add_argument('--resize', metavar='-r', type=str,
                            help='if you are in rush resize the image by a fraction e.q. 1/5 etc. \n if you do not define it it will not resize the images')
    
    parser.add_argument('--type', metavar='-t', type=str,
                            help= ' the type of generation of similarity matrix. takes 'mse', 'corrcoef' or 'ssim'. Default is 'ssim')
    
    args = parser.parse_args()
    
    app = __app__(args)
    
    #app = __app__()