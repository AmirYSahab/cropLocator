# cropLocator
reads images and prints out the 4 corner coordinates of the croped region

            _________________________________________________________________________________________

__author__= ['Amir Yeganehsahab']

__url__ = (' ')

__version__= '0.0'

__doc__= 

            reads images and prints out the 4 corner coordinates of the croped region
            
            inputs: two images (one is a rotated/nonrotated crop of the base image) 
            
            output: coordinates of the most ptobable croped region
            
__inputs__ = 

            base_image_path: str
            
            croped_image_path: str
            
            resize: float or None
            
            type: str (one of 'corrcoef','ssim' or 'mse')
            
__outputs__ = 

            coordinates of the most probable croped region 
            
            including: 
            
                upper left corner  (ulc): tuple
                
                upper right corner (urc): tuple
                
                lower lef corner   (llc): tuple
                
                lower right corner (lrc): tuple
            
            _________________________________________________________________________________________

example run: 

             python3 main.py --StarMap=~/Pictures/inputs/StarMap.png --crop=~/Pictures/inputs/Small_area.png  --resize=1/5 --type='ssim'

example output: 

                        upper left corner:  (117, 257)
                        
                        upper right corner:  (231, 257)
                        
                        lowe left corner:  (117, 371)
                        
                        lower right corner:  (231, 371)
                        
