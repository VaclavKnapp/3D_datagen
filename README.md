# 3D_datagen

## Clone this repo
'''bash
git clone https://github.com/VaclavKnapp/3D_datagen.git
'''

## Infinigen
1. Clone the infinigen repo
   '''bash
   git clone https://github.com/princeton-vl/infinigen.git
   '''
2. move the code from the 3D_datagen/infigen folder in the infinigen repo folder
   '''bash
   mv 3D_datagen/infinigen/gen_assests_blend.py infinigen/infinigen_examples
   mv 3D_datagen/infinigen/oddity.py infinigen/.
   mv 3D_datagen/infinigen/render.sh infinigen/.
   '''
### Generating images
1. Generating .blend files
   '''bash
   cd infinigen
   '''
   '''bash
   python infinigen_examples/gen_assests_blend.py -f 3D_datagen/infinigen/factories.txt -o 3D_datagen/infinigen_blend -n 10 --texture_folder 3D_datagen/textures
   '''
2. Rendering of images
   '''bash
   bash infinigen/render.sh 3D_datagen/infinigen_blend 3D_datagen/infinigen_images 3D_datagen/backgrounds
   '''
### Running oddity detections
1. Add path with the generated images (in this case '3D_datagen/infinigen_images' on line '203' in 'infinigen/oddity.py'
2. start the code
   '''bash
   python infinigen/oddity.py
   '''

## Zeroverse (my implementation)
1. Change your output folder in 'generate_sets.sh' 
2. Generate .blend file
   '''bash
   bash 3D_datagen/zeroverse/generate_sets.sh
   '''
3. Render images
   '''bash
   bash 3D_datagen/zeroverse/render.sh YOUR_BLEND_FOLDER 3D_datagen/zeroverse_images 3D_datagen/backgrounds
   '''
### Running oddity detections
1. Add path with the generated images (in this case '3D_datagen/zeroverse_images' on line '258' in 'zeroverse/oddity.py'
2. start the code
   '''bash
   python infinigen/oddity.py
   ''' 

## Shapegen
1.
